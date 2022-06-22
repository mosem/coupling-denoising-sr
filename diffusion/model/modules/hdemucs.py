# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
This code contains the spectrogram and Hybrid version of Demucs.
"""
from copy import deepcopy
import math
import typing as tp

from openunmix.filtering import wiener
import torch
from torch import nn
from torch.nn import functional as F

from diffusion.model.modules.utils import capture_init, unfold
from diffusion.model.modules.spec import spectro, ispectro

from torchaudio.functional import resample



def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class BLSTM(nn.Module):
    """
    BiLSTM with same hidden units as input dim.
    If `max_steps` is not None, input will be splitting in overlapping
    chunks and the LSTM applied separately on each chunk.
    """

    def __init__(self, dim, layers=1, max_steps=None, skip=False):
        super().__init__()
        assert max_steps is None or max_steps % 4 == 0
        self.max_steps = max_steps
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)
        self.skip = skip

    def forward(self, x):
        B, C, T = x.shape
        y = x
        framed = False
        if self.max_steps is not None and T > self.max_steps:
            width = self.max_steps
            stride = width // 2
            frames = unfold(x, width, stride)
            nframes = frames.shape[2]
            framed = True
            x = frames.permute(0, 2, 1, 3).reshape(-1, C, width)

        x = x.permute(2, 0, 1)

        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        if framed:
            out = []
            frames = x.reshape(B, -1, C, width)
            limit = stride // 2
            for k in range(nframes):
                if k == 0:
                    out.append(frames[:, k, :, :-limit])
                elif k == nframes - 1:
                    out.append(frames[:, k, :, limit:])
                else:
                    out.append(frames[:, k, :, limit:-limit])
            out = torch.cat(out, -1)
            out = out[..., :T]
            x = out
        if self.skip:
            x = x + y
        return x


class LocalState(nn.Module):
    """Local state allows to have attention based only on data (no positional embedding),
    but while setting a constraint on the time window (e.g. decaying penalty term).
    Also a failed experiments with trying to provide some frequency based attention.
    """

    def __init__(self, channels: int, heads: int = 4, nfreqs: int = 0, ndecay: int = 4):
        super().__init__()
        assert channels % heads == 0, (channels, heads)
        self.heads = heads
        self.nfreqs = nfreqs
        self.ndecay = ndecay
        self.content = nn.Conv1d(channels, channels, 1)
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        if nfreqs:
            self.query_freqs = nn.Conv1d(channels, heads * nfreqs, 1)
        if ndecay:
            self.query_decay = nn.Conv1d(channels, heads * ndecay, 1)
            # Initialize decay close to zero (there is a sigmoid), for maximum initial window.
            self.query_decay.weight.data *= 0.01
            assert self.query_decay.bias is not None  # stupid type checker
            self.query_decay.bias.data[:] = -2
        self.proj = nn.Conv1d(channels + heads * nfreqs, channels, 1)

    def forward(self, x):
        B, C, T = x.shape
        heads = self.heads
        indexes = torch.arange(T, device=x.device, dtype=x.dtype)
        # left index are keys, right index are queries
        delta = indexes[:, None] - indexes[None, :]

        queries = self.query(x).view(B, heads, -1, T)
        keys = self.key(x).view(B, heads, -1, T)
        # t are keys, s are queries
        dots = torch.einsum("bhct,bhcs->bhts", keys, queries)
        dots /= keys.shape[2] ** 0.5
        if self.nfreqs:
            periods = torch.arange(1, self.nfreqs + 1, device=x.device, dtype=x.dtype)
            freq_kernel = torch.cos(2 * math.pi * delta / periods.view(-1, 1, 1))
            freq_q = self.query_freqs(x).view(B, heads, -1, T) / self.nfreqs ** 0.5
            dots += torch.einsum("fts,bhfs->bhts", freq_kernel, freq_q)
        if self.ndecay:
            decays = torch.arange(1, self.ndecay + 1, device=x.device, dtype=x.dtype)
            decay_q = self.query_decay(x).view(B, heads, -1, T)
            decay_q = torch.sigmoid(decay_q) / 2
            decay_kernel = - decays.view(-1, 1, 1) * delta.abs() / self.ndecay ** 0.5
            dots += torch.einsum("fts,bhfs->bhts", decay_kernel, decay_q)

        # Kill self reference.
        dots.masked_fill_(torch.eye(T, device=dots.device, dtype=torch.bool), -100)
        weights = torch.softmax(dots, dim=2)

        content = self.content(x).view(B, heads, -1, T)
        result = torch.einsum("bhts,bhct->bhcs", weights, content)
        if self.nfreqs:
            time_sig = torch.einsum("bhts,fts->bhfs", weights, freq_kernel)
            result = torch.cat([result, time_sig], 2)
        result = result.reshape(B, -1, T)
        return x + self.proj(result)


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 0):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x):
        return self.scale[:, None] * x


class DConv(nn.Module):
    """
    New residual branches in each encoder layer.
    This alternates dilated convolutions, potentially with LSTMs and attention.
    Also before entering each residual branch, dimension is projected on a smaller subspace,
    e.g. of dim `channels // compress`.
    """

    def __init__(self, channels: int, compress: float = 4, depth: int = 2, init: float = 1e-4,
                 norm=True, attn=False, heads=4, ndecay=4, lstm=False, gelu=True,
                 kernel=3, dilate=True):
        """
        Args:
            channels: input/output channels for residual branch.
            compress: amount of channel compression inside the branch.
            depth: number of layers in the residual branch. Each layer has its own
                projection, and potentially LSTM and attention.
            init: initial scale for LayerNorm.
            norm: use GroupNorm.
            attn: use LocalAttention.
            heads: number of heads for the LocalAttention.
            ndecay: number of decay controls in the LocalAttention.
            lstm: use LSTM.
            gelu: Use GELU activation.
            kernel: kernel size for the (dilated) convolutions.
            dilate: if true, use dilation, increasing with the depth.
        """

        super().__init__()
        assert kernel % 2 == 1
        self.channels = channels
        self.compress = compress
        self.depth = abs(depth)
        dilate = depth > 0

        norm_fn: tp.Callable[[int], nn.Module]
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(1, d)  # noqa

        hidden = int(channels / compress)

        act: tp.Type[nn.Module]
        if gelu:
            act = nn.GELU
        else:
            act = nn.ReLU

        self.layers = nn.ModuleList([])
        for d in range(self.depth):
            dilation = 2 ** d if dilate else 1
            padding = dilation * (kernel // 2)
            mods = [
                nn.Conv1d(channels, hidden, kernel, dilation=dilation, padding=padding),
                norm_fn(hidden), act(),
                nn.Conv1d(hidden, 2 * channels, 1),
                norm_fn(2 * channels), nn.GLU(1),
                LayerScale(channels, init),
            ]
            if attn:
                mods.insert(3, LocalState(hidden, heads=heads, ndecay=ndecay))
            if lstm:
                mods.insert(3, BLSTM(hidden, layers=2, max_steps=200, skip=True))
            layer = nn.Sequential(*mods)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False, freq=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.freq = freq
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            if self.freq:
                embedding = self.noise_func(noise_embed).view(batch, -1, 1, 1)
            else:
                embedding = self.noise_func(noise_embed).view(batch, -1, 1)
            x = x + embedding
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ScaledEmbedding(nn.Module):
    """
    Boost learning rate for embeddings (with `scale`).
    Also, can make embeddings continuous with `smooth`.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 scale: float = 10., smooth=False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if smooth:
            weight = torch.cumsum(self.embedding.weight.data, dim=0)
            # when summing gaussian, overscale raises as sqrt(n), so we nornalize by that.
            weight = weight / torch.arange(1, num_embeddings + 1).to(weight).sqrt()[:, None]
            self.embedding.weight.data[:] = weight
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self):
        return self.embedding.weight * self.scale

    def forward(self, x):
        out = self.embedding(x) * self.scale
        return out


class HEncLayer(nn.Module):
    def __init__(self, chin, chout, noise_level_emb_dim=None, kernel_size=8, stride=4, norm_groups=1, empty=False,
                 freq=True, dconv=True, norm=True, context=0, dconv_kw={}, pad=True,
                 rewrite=True):
        """Encoder layer. This used both by the time and the frequency branch.

        Args:
            chin: number of input channels.
            chout: number of output channels.
            norm_groups: number of groups for group norm.
            empty: used to make a layer with just the first conv. this is used
                before merging the time and freq. branches.
            freq: this is acting on frequencies.
            dconv: insert DConv residual branches.
            norm: use GroupNorm.
            context: context size for the 1x1 conv.
            dconv_kw: list of kwargs for the DConv class.
            pad: pad the input. Padding is done so that the output size is
                always the input size / stride.
            rewrite: add 1x1 conv at the end of the layer.
        """
        super().__init__()
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa
        if pad:
            pad = kernel_size // 4
        else:
            pad = 0
        klass = nn.Conv1d
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.norm = norm
        self.pad = pad

        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, chout, freq=freq)

        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            pad = [pad, 0]
            klass = nn.Conv2d
        self.conv = klass(chin, chout, kernel_size, stride, pad)
        if self.empty:
            return
        self.norm1 = norm_fn(chout)
        self.rewrite = None
        if rewrite:
            self.rewrite = klass(chout, 2 * chout, 1 + 2 * context, 1, context)
            self.norm2 = norm_fn(2 * chout)

        self.dconv = None
        if dconv:
            self.dconv = DConv(chout, **dconv_kw)

    def forward(self, x, time_emb, inject=None):
        """
        `inject` is used to inject the result from the time branch into the frequency branch,
        when both have the same stride.
        """
        if not self.freq and x.dim() == 4:
            B, C, Fr, T = x.shape
            x = x.view(B, -1, T)

        if not self.freq:
            le = x.shape[-1]
            if not le % self.stride == 0:
                x = F.pad(x, (0, self.stride - (le % self.stride)))
        y = self.conv(x)

        if self.empty:
            return y

        y = self.noise_func(y, time_emb)  # TODO: this is where I inject the time-embedding, projecting it to the dimension of chout. Is this correct?

        if inject is not None:
            assert inject.shape[-1] == y.shape[-1], (inject.shape, y.shape)
            if inject.dim() == 3 and y.dim() == 4:
                inject = inject[:, :, None]
            y = y + inject
        y = F.gelu(self.norm1(y))
        if self.dconv:
            if self.freq:
                B, C, Fr, T = y.shape
                y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
            y = self.dconv(y)
            if self.freq:
                y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        if self.rewrite:
            z = self.norm2(self.rewrite(y))
            z = F.glu(z, dim=1)
        else:
            z = y
        return z


class MultiWrap(nn.Module):
    """
    Takes one layer and replicate it N times. each replica will act
    on a frequency band. All is done so that if the N replica have the same weights,
    then this is exactly equivalent to applying the original module on all frequencies.

    This is a bit over-engineered to avoid edge artifacts when splitting
    the frequency bands, but it is possible the naive implementation would work as well...
    """

    def __init__(self, layer, split_ratios):
        """
        Args:
            layer: module to clone, must be either HEncLayer or HDecLayer.
            split_ratios: list of float indicating which ratio to keep for each band.
        """
        super().__init__()
        self.split_ratios = split_ratios
        self.layers = nn.ModuleList()
        self.conv = isinstance(layer, HEncLayer)
        assert not layer.norm
        assert layer.freq
        assert layer.pad
        if not self.conv:
            assert not layer.context_freq
        for k in range(len(split_ratios) + 1):
            lay = deepcopy(layer)
            if self.conv:
                lay.conv.padding = (0, 0)
            else:
                lay.pad = False
            for m in lay.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            self.layers.append(lay)

    def forward(self, x, skip=None, length=None):
        B, C, Fr, T = x.shape

        ratios = list(self.split_ratios) + [1]
        start = 0
        outs = []
        for ratio, layer in zip(ratios, self.layers):
            if self.conv:
                pad = layer.kernel_size // 4
                if ratio == 1:
                    limit = Fr
                    frames = -1
                else:
                    limit = int(round(Fr * ratio))
                    le = limit - start
                    if start == 0:
                        le += pad
                    frames = round((le - layer.kernel_size) / layer.stride + 1)
                    limit = start + (frames - 1) * layer.stride + layer.kernel_size
                    if start == 0:
                        limit -= pad
                assert limit - start > 0, (limit, start)
                assert limit <= Fr, (limit, Fr)
                y = x[:, :, start:limit, :]
                if start == 0:
                    y = F.pad(y, (0, 0, pad, 0))
                if ratio == 1:
                    y = F.pad(y, (0, 0, 0, pad))
                outs.append(layer(y))
                start = limit - layer.kernel_size + layer.stride
            else:
                if ratio == 1:
                    limit = Fr
                else:
                    limit = int(round(Fr * ratio))
                last = layer.last
                layer.last = True

                y = x[:, :, start:limit]
                s = skip[:, :, start:limit]
                out, _ = layer(y, s, None)
                if outs:
                    outs[-1][:, :, -layer.stride:] += (
                            out[:, :, :layer.stride] - layer.conv_tr.bias.view(1, -1, 1, 1))
                    out = out[:, :, layer.stride:]
                if ratio == 1:
                    out = out[:, :, :-layer.stride // 2, :]
                if start == 0:
                    out = out[:, :, layer.stride // 2:, :]
                outs.append(out)
                layer.last = last
                start = limit
        out = torch.cat(outs, dim=2)
        if not self.conv and not last:
            out = F.gelu(out)
        if self.conv:
            return out
        else:
            return out, None


class HDecLayer(nn.Module):
    def __init__(self, chin, chout, noise_level_emb_dim=None, last=False, kernel_size=8, stride=4, norm_groups=1, empty=False,
                 freq=True, dconv=True, norm=True, context=1, dconv_kw={}, pad=True,
                 context_freq=True, rewrite=True):
        """
        Same as HEncLayer but for decoder. See `HEncLayer` for documentation.
        """
        super().__init__()
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa
        if pad:
            pad = kernel_size // 4
        else:
            pad = 0
        self.pad = pad
        self.last = last
        self.freq = freq
        self.chin = chin
        self.empty = empty
        self.stride = stride
        self.kernel_size = kernel_size
        self.norm = norm
        self.context_freq = context_freq

        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, chin, freq=freq)

        klass = nn.Conv1d
        klass_tr = nn.ConvTranspose1d
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            klass = nn.Conv2d
            klass_tr = nn.ConvTranspose2d
        self.conv_tr = klass_tr(chin, chout, kernel_size, stride)
        self.norm2 = norm_fn(chout)
        if self.empty:
            return
        self.rewrite = None
        if rewrite:
            if context_freq:
                self.rewrite = klass(chin, 2 * chin, 1 + 2 * context, 1, context)
            else:
                self.rewrite = klass(chin, 2 * chin, [1, 1 + 2 * context], 1,
                                     [0, context])
            self.norm1 = norm_fn(2 * chin)

        self.dconv = None
        if dconv:
            self.dconv = DConv(chin, **dconv_kw)

    def forward(self, x, time_emb, skip, length):
        if self.freq and x.dim() == 3:
            B, C, T = x.shape
            x = x.view(B, self.chin, -1, T)

        if not self.empty:
            x = x + skip

            x = self.noise_func(x, time_emb) # TODO: this is where I inject the time-embedding, projecting it to the dimension of chin. Is this correct?

            if self.rewrite:
                y = F.glu(self.norm1(self.rewrite(x)), dim=1)
            else:
                y = x
            if self.dconv:
                if self.freq:
                    B, C, Fr, T = y.shape
                    y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
                y = self.dconv(y)
                if self.freq:
                    y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        else:
            y = x
            assert skip is None
        z = self.norm2(self.conv_tr(y))
        if self.freq:
            if self.pad:
                z = z[..., self.pad:-self.pad, :]
        else:
            z = z[..., self.pad:self.pad + length]
            assert z.shape[-1] == length, (z.shape[-1], length)
        if not self.last:
            z = F.gelu(z)
        return z, y


import logging
logger = logging.getLogger('base')

class HDemucs(nn.Module):
    """
    Spectrogram and hybrid Demucs model.
    The spectrogram model has the same structure as Demucs, except the first few layers are over the
    frequency axis, until there is only 1 frequency, and then it moves to time convolutions.
    Frequency layers can still access information across time steps thanks to the DConv residual.

    Hybrid model have a parallel time branch. At some layer, the time branch has the same stride
    as the frequency branch and then the two are combined. The opposite happens in the decoder.

    Models can either use naive iSTFT from masking, Wiener filtering ([Ulhih et al. 2017]),
    or complex as channels (CaC) [Choi et al. 2020]. Wiener filtering is based on
    Open Unmix implementation [Stoter et al. 2019].

    The loss is always on the temporal domain, by backpropagating through the above
    output methods and iSTFT. This allows to define hybrid models nicely. However, this breaks
    a bit Wiener filtering, as doing more iteration at test time will change the spectrogram
    contribution, without changing the one from the waveform, which will lead to worse performance.
    I tried using the residual option in OpenUnmix Wiener implementation, but it didn't improve.
    CaC on the other hand provides similar performance for hybrid, and works naturally with
    hybrid models.

    This model also uses frequency embeddings are used to improve efficiency on convolutions
    over the freq. axis, following [Isik et al. 2020] (https://arxiv.org/pdf/2008.04470.pdf).

    Unlike classic Demucs, there is no resampling here, and normalization is always applied.
    """

    @capture_init
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 channels=48,
                 channels_time=None,
                 growth=2,
                 # STFT
                 nfft=4096,
                 wiener_iters=0,
                 end_iters=0,
                 wiener_residual=False,
                 cac=True,
                 # Main structure
                 depth=6,
                 rewrite=True,
                 hybrid=True,
                 hybrid_old=False,
                 # Frequency branch
                 multi_freqs=None,
                 multi_freqs_depth=2,
                 freq_emb=0.2,
                 emb_scale=10,
                 emb_smooth=True,
                 # Convolutions
                 kernel_size=8,
                 time_stride=2,
                 stride=4,
                 context=1,
                 context_enc=0,
                 # Normalization
                 norm_starts=4,
                 norm_groups=4,
                 # DConv residual branch
                 dconv_mode=1,
                 dconv_depth=2,
                 dconv_comp=4,
                 dconv_attn=4,
                 dconv_lstm=4,
                 dconv_init=1e-4,
                 # Weight init
                 rescale=0.1,
                 # Metadata
                 samplerate=44100,
                 segment=4 * 10,
                 lr_sr=16000,
                 hr_sr=16000,
                 upsample=False):
        """
        Args:
            sources (list[str]): list of source names.
            audio_channels (int): input/output audio channels.
            channels (int): initial number of hidden channels.
            channels_time: if not None, use a different `channels` value for the time branch.
            growth: increase the number of hidden channels by this factor at each layer.
            nfft: number of fft bins. Note that changing this require careful computation of
                various shape parameters and will not work out of the box for hybrid models.
            wiener_iters: when using Wiener filtering, number of iterations at test time.
            end_iters: same but at train time. For a hybrid model, must be equal to `wiener_iters`.
            wiener_residual: add residual source before wiener filtering.
            cac: uses complex as channels, i.e. complex numbers are 2 channels each
                in input and output. no further processing is done before ISTFT.
            depth (int): number of layers in the encoder and in the decoder.
            rewrite (bool): add 1x1 convolution to each layer.
            hybrid (bool): make a hybrid time/frequency domain, otherwise frequency only.
            hybrid_old: some models trained for MDX had a padding bug. This replicates
                this bug to avoid retraining them.
            multi_freqs: list of frequency ratios for splitting frequency bands with `MultiWrap`.
            multi_freqs_depth: how many layers to wrap with `MultiWrap`. Only the outermost
                layers will be wrapped.
            freq_emb: add frequency embedding after the first frequency layer if > 0,
                the actual value controls the weight of the embedding.
            emb_scale: equivalent to scaling the embedding learning rate
            emb_smooth: initialize the embedding with a smooth one (with respect to frequencies).
            kernel_size: kernel_size for encoder and decoder layers.
            stride: stride for encoder and decoder layers.
            time_stride: stride for the final time layer, after the merge.
            context: context for 1x1 conv in the decoder.
            context_enc: context for 1x1 conv in the encoder.
            norm_starts: layer at which group norm starts being used.
                decoder layers are numbered in reverse order.
            norm_groups: number of groups for group norm.
            dconv_mode: if 1: dconv in encoder only, 2: decoder only, 3: both.
            dconv_depth: depth of residual DConv branch.
            dconv_comp: compression of DConv branch.
            dconv_attn: adds attention layers in DConv branch starting at this layer.
            dconv_lstm: adds a LSTM layer in DConv branch starting at this layer.
            dconv_init: initial scale for the DConv branch LayerScale.
            rescale: weight recaling trick

        """
        super().__init__()
        self.cac = cac
        self.wiener_residual = wiener_residual
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.channels = channels
        self.samplerate = samplerate
        self.segment = segment

        self.nfft = nfft
        self.hop_length = nfft // 4
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters
        self.freq_emb = None
        self.hybrid = hybrid
        self.hybrid_old = hybrid_old
        if hybrid_old:
            assert hybrid, "hybrid_old must come with hybrid=True"
        if hybrid:
            assert wiener_iters == end_iters

        self.lr_sr = lr_sr
        self.hr_sr = hr_sr
        self.scale_factor = int(self.hr_sr / self.lr_sr)
        self.upsample = upsample

        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(self.channels),
            nn.Linear(self.channels, self.channels * 4),
            Swish(),
            nn.Linear(self.channels * 4, self.channels)
        )

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        if hybrid:
            self.tencoder = nn.ModuleList()
            self.tdecoder = nn.ModuleList()

        chin = self.in_channels
        chin_z = chin  # number of channels for the freq branch
        if self.cac:
            chin_z *= 2
        chout = channels_time or channels
        chout_z = channels
        freqs = nfft // 2

        for index in range(depth):
            lstm = index >= dconv_lstm
            attn = index >= dconv_attn
            norm = index >= norm_starts
            freq = freqs > 1
            stri = stride
            ker = kernel_size
            if not freq:
                assert freqs == 1
                ker = time_stride * 2
                stri = time_stride

            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            kw = {
                'kernel_size': ker,
                'stride': stri,
                'freq': freq,
                'pad': pad,
                'norm': norm,
                'rewrite': rewrite,
                'norm_groups': norm_groups,
                'dconv_kw': {
                    'lstm': lstm,
                    'attn': attn,
                    'depth': dconv_depth,
                    'compress': dconv_comp,
                    'init': dconv_init,
                    'gelu': True,
                }
            }
            kwt = dict(kw)
            kwt['freq'] = 0
            kwt['kernel_size'] = kernel_size
            kwt['stride'] = stride
            kwt['pad'] = True
            kw_dec = dict(kw)
            multi = False
            if multi_freqs and index < multi_freqs_depth:
                multi = True
                kw_dec['context_freq'] = False

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            enc = HEncLayer(chin_z, chout_z, noise_level_emb_dim=channels,
                            dconv=dconv_mode & 1, context=context_enc, **kw)
            if hybrid and freq:
                tenc = HEncLayer(chin, chout, noise_level_emb_dim=channels, dconv=dconv_mode & 1, context=context_enc,
                                 empty=last_freq, **kwt)
                self.tencoder.append(tenc)

            if multi:
                enc = MultiWrap(enc, multi_freqs)
            self.encoder.append(enc)
            if index == 0:
                chin = self.out_channels
                chin_z = chin
                if self.cac:
                    chin_z *= 2
            dec = HDecLayer(chout_z, chin_z, noise_level_emb_dim=channels, dconv=dconv_mode & 2,
                            last=index == 0, context=context, **kw_dec)
            if multi:
                dec = MultiWrap(dec, multi_freqs)
            if hybrid and freq:
                tdec = HDecLayer(chout, chin, noise_level_emb_dim=channels, dconv=dconv_mode & 2, empty=last_freq,
                                 last=index == 0, context=context, **kwt)
                self.tdecoder.insert(0, tdec)
            self.decoder.insert(0, dec)

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)
            if freq:
                if freqs <= kernel_size:
                    freqs = 1
                else:
                    freqs //= stride
            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(
                    freqs, chin_z, smooth=emb_smooth, scale=emb_scale)
                self.freq_emb_scale = freq_emb

        if rescale:
            rescale_module(self, reference=rescale)

    def _spec(self, x):
        hl = self.hop_length
        nfft = self.nfft
        x0 = x  # noqa

        if self.hybrid:
            # We re-pad the signal in order to keep the property
            # that the size of the output is exactly the size of the input
            # divided by the stride (here hop_length), when divisible.
            # This is achieved by padding by 1/4th of the kernel size (here nfft).
            # which is not supported by torch.stft.
            # Having all convolution operations follow this convention allow to easily
            # align the time and frequency branches later on.
            assert hl == nfft // 4
            le = int(math.ceil(x.shape[-1] / hl))
            pad = hl // 2 * 3
            if not self.hybrid_old:
                x = F.pad(x, (pad, pad + le * hl - x.shape[-1]), mode='reflect')
            else:
                x = F.pad(x, (pad, pad + le * hl - x.shape[-1]))

        z = spectro(x, nfft, hl)[..., :-1, :]
        if self.hybrid:
            assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
            z = z[..., 2:2 + le]
        return z

    def _ispec(self, z, length=None, scale=0):
        hl = self.hop_length // (4 ** scale)
        z = F.pad(z, (0, 0, 0, 1))
        if self.hybrid:
            z = F.pad(z, (2, 2))
            pad = hl // 2 * 3
            if not self.hybrid_old:
                le = hl * int(math.ceil(length / hl)) + 2 * pad
            else:
                le = hl * int(math.ceil(length / hl))
            x = ispectro(z, hl, length=le)
            if not self.hybrid_old:
                x = x[..., pad:pad + length]
            else:
                x = x[..., :length]
        else:
            x = ispectro(z, hl, length)
        return x

    def _magnitude(self, z):
        # return the magnitude of the spectrogram, except when cac is True,
        # in which case we just move the complex dimension to the channel one.
        if self.cac:
            B, C, Fr, T = z.shape
            m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
            m = m.reshape(B, C * 2, Fr, T)
        else:
            m = z.abs()
        return m

    def _mask(self, z, m):
        # Apply masking given the mixture spectrogram `z` and the estimated mask `m`.
        # If `cac` is True, `m` is actually a full spectrogram and `z` is ignored.
        niters = self.wiener_iters
        if self.cac:
            B, S, C, Fr, T = m.shape
            out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
            out = torch.view_as_complex(out.contiguous())
            return out
        if self.training:
            niters = self.end_iters
        if niters < 0:
            z = z[:, None]
            return z / (1e-8 + z.abs()) * m
        else:
            return self._wiener(m, z, niters)

    def _wiener(self, mag_out, mix_stft, niters):
        # apply wiener filtering from OpenUnmix.
        init = mix_stft.dtype
        wiener_win_len = 300
        residual = self.wiener_residual

        B, S, C, Fq, T = mag_out.shape
        mag_out = mag_out.permute(0, 4, 3, 2, 1)
        mix_stft = torch.view_as_real(mix_stft.permute(0, 3, 2, 1))

        outs = []
        for sample in range(B):
            pos = 0
            out = []
            for pos in range(0, T, wiener_win_len):
                frame = slice(pos, pos + wiener_win_len)
                z_out = wiener(
                    mag_out[sample, frame], mix_stft[sample, frame], niters,
                    residual=residual)
                out.append(z_out.transpose(-1, -2))
            outs.append(torch.cat(out, dim=0))
        out = torch.view_as_complex(torch.stack(outs, 0))
        out = out.permute(0, 4, 3, 2, 1).contiguous()
        if residual:
            out = out[:, :-1]
        assert list(out.shape) == [B, S, C, Fq, T]
        return out.to(init)

    def forward(self, mix, time):
        t = self.noise_level_mlp(time)

        if self.upsample:
            mix = resample(mix, self.lr_sr, self.hr_sr)
        x = mix
        # logger.info(f'hdemucs in shape: {x.shape}')
        length = x.shape[-1]

        z = self._spec(mix)
        mag = self._magnitude(z)
        x = mag

        B, C, Fq, T = x.shape

        # unlike previous Demucs, we always normalize because it is easier.
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        # x will be the freq. branch input.

        if self.hybrid:
            # Prepare the time branch input.
            xt = mix
            meant = xt.mean(dim=(1, 2), keepdim=True)
            stdt = xt.std(dim=(1, 2), keepdim=True)
            xt = (xt - meant) / (1e-5 + stdt)

        # okay, this is a giant mess I know...
        saved = []  # skip connections, freq.
        saved_t = []  # skip connections, time.
        lengths = []  # saved lengths to properly remove padding, freq branch.
        lengths_t = []  # saved lengths for time branch.
        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if self.hybrid and idx < len(self.tencoder):
                # we have not yet merged branches.
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt, t)
                if not tenc.empty:
                    # save for skip connection
                    saved_t.append(xt)
                else:
                    # tenc contains just the first conv., so that now time and freq.
                    # branches have the same shape and can be merged.
                    inject = xt
            x = encode(x, t, inject=inject)
            # logger.info(f'encode-{idx}: x: {x.shape}, xt: {xt.shape if idx < len(self.tencoder) else None}')
            if idx == 0 and self.freq_emb is not None:
                # add frequency embedding to allow for non equivariant convolutions
                # over the frequency axis.
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb

            saved.append(x)

        x = torch.zeros_like(x)
        if self.hybrid:
            xt = torch.zeros_like(x)
        # initialize everything to zero (signal will go through u-net skips).

        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, t, skip, lengths.pop(-1))
            # `pre` contains the output just before final transposed convolution,
            # which is used when the freq. and time branch separate.

            if self.hybrid:
                offset = self.depth - len(self.tdecoder)
            if self.hybrid and idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    assert pre.shape[2] == 1, pre.shape
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, t, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, t, skip, length_t)
            # logger.info(f'decode-{idx}: x: {x.shape}, xt: {xt.shape if idx >= offset else None}, pre: {pre.shape}')
        # Let's make sure we used all stored skip connections.
        assert len(saved) == 0
        assert len(lengths_t) == 0
        assert len(saved_t) == 0

        # logger.info(f'hdemucs pre view shape: {x.shape}')

        # S = len(self.sources)
        x = x.view(B, self.out_channels, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        # logger.info(f'hdemucs post view x shape: {x.shape}')
        # logger.info(f'hdemucs post view z shape: {z.shape}')

        zout = self._mask(z, x)
        # logger.info(f'hdemucs pre ispec zout shape: {zout.shape}')
        x = self._ispec(zout, length)

        # logger.info(f'hdemucs post ispec x shape: {x.shape}')

        if self.hybrid:
            # logger.info(f'hdemucs pre view xt shape: {xt.shape}')
            xt = xt.view(B, self.out_channels, -1, length)
            # logger.info(f'hdemucs post view xt shape: {xt.shape}')
            xt = xt * stdt[:, None] + meant[:, None]
            x = xt + x

        x = x.squeeze(dim=2)

        # logger.info(f'hdemucs out shape: {x.shape}')
        return x