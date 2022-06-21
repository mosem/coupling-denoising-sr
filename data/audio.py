# This code is based on FAIR's Demucs denoiser: https://github.com/facebookresearch/denoiser

import math
import torchaudio
from torch.nn import functional as F
from pathlib import Path


class Audioset:
    def __init__(self, files=None, length=None, stride=None,
                 pad=True, sample_rate=None,
                 channels=None):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.file_lengths = []
        self.length = length
        self.stride = stride or length
        self.sample_rate = sample_rate
        self.channels = channels

        for file, file_length in self.files:
            if length is None:
                examples = 1
                self.file_lengths.append(file_length)
            elif file_length < length:
                examples = 1 if pad else 0
                if pad:
                    self.file_lengths.append(file_length)
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
                self.file_lengths.extend([self.length]*examples)
            else:
                examples = (file_length - self.length) // self.stride + 1
                self.file_lengths.extend([self.length] * examples)
            self.num_examples.append(examples)


    def get_file_lengths(self):
        return self.file_lengths

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = 0
            offset = 0
            if self.length is not None:
                offset = self.stride * index
                num_frames = self.length
            if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
                out, sr = torchaudio.load(str(file),
                                          frame_offset=offset,
                                          num_frames=num_frames or -1)
            else:
                out, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)

            if sr != self.sample_rate:
                raise RuntimeError(f"Expected {file} to have sample rate of "
                                   f"{self.sample_rate}, but got {sr}")
            if out.shape[0] != self.channels:
                raise RuntimeError(f"Expected {file} to have shape of "
                                   f"{out.shape[0]}, but got {self.channels}")
            if num_frames:
                out = F.pad(out, (0, num_frames - out.shape[-1]))

            return out, Path(file).stem