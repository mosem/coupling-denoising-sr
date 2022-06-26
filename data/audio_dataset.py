from torch.utils.data import Dataset
import os
import json
import math

from torch.nn import functional as F

from .audio import Audioset

import logging
logger = logging.getLogger('base')

def match_target_to_source_length(target_sig, source_sig):
    target_len = target_sig.shape[-1]
    source_len = source_sig.shape[-1]
    if target_len < source_len:
        target_sig = F.pad(target_sig, (0, source_len - target_sig.shape[-1]))
    elif target_len > source_len:
        target_sig = target_sig[..., :source_len]
    return target_sig


def compute_output_length(length, depth):
    for i in range(depth - 1):
        length = math.floor((length - 1) / 2 + 1)
    for j in range(depth - 1):
        length = math.floor(length * 2)
    return length


class AudioDataset(Dataset):

    def __init__(self, json_dir, source_sr=8000, target_sr=16000, stride=None, segment=None, pad=True,
                 pad_to_output_length=False, data_len=-1, need_source_raw=False):
        self.data_len = data_len
        self.need_source_raw = need_source_raw
        self.pad_to_output_length = pad_to_output_length

        self.source_sr = source_sr
        self.target_sr = target_sr

        source_stride = stride * source_sr if stride else None
        target_stride = stride * target_sr if stride else None
        source_length = segment * source_sr if segment else None
        target_length = segment * target_sr if segment else None

        target_json = os.path.join(json_dir, 'target.json')
        with open(target_json, 'r') as f:
            target = json.load(f)
        target.sort()
        self.target_set = Audioset(target, sample_rate=target_sr, length=target_length, stride=target_stride, pad=pad, channels=1)


        source_json = os.path.join(json_dir, 'source_up.json') if source_sr != target_sr \
                                                                    else os.path.join(json_dir,'source.json')
        with open(source_json, 'r') as f:
            source = json.load(f)
        source.sort()
        self.source_set = Audioset(source, sample_rate=target_sr, length=target_length, stride=target_stride, pad=pad, channels=1)
        assert len(self.target_set) == len(self.source_set)

        self.dataset_len = len(self.target_set)

        # if self.need_source_raw:
        #     source_raw_json = os.path.join(json_dir, 'source.json')
        #     with open(source_raw_json, 'r') as f:
        #         source_raw = json.load(f)
        #     self.source_raw_set = Audioset(source_raw, sample_rate=source_sr, length=source_length, stride=source_stride, pad=pad, channels=1)

        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)


    def get_file_lengths(self):
        return self.target_set.get_file_lengths()

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        target_sig, target_filename = self.target_set[index]
        source_sig, source_filename = self.source_set[index]

        target_sig = match_target_to_source_length(target_sig, source_sig)

        target_len = target_sig.shape[-1]

        if self.pad_to_output_length:
            sig_len = compute_output_length(target_sig.shape[-1], 5)
            target_sig = F.pad(target_sig, (0, sig_len - target_sig.shape[-1]))
            source_sig = F.pad(source_sig, (0, sig_len - source_sig.shape[-1]))

        return {'target': target_sig, 'source': source_sig, 'filename': target_filename, 'length': target_len}

        # if self.need_source_raw:
        #     source_raw_sig, source_raw_filename = self.source_raw_set[index]
        #     # augment?
        #     return {'source_raw': source_raw_sig, 'target': target_sig, 'source': source_sig, 'filename': target_filename, 'length': target_len}
        # else:
        #     # augment?
        #     return {'target': target_sig, 'source': source_sig, 'filename': target_filename, 'length': target_len}