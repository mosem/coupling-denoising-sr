# This code is from FAIR's Demucs denoiser: https://github.com/facebookresearch/denoiser

from pathlib import Path
import os
import torchaudio
from collections import namedtuple
import json
import sys

Info = namedtuple("Info", ["length", "sample_rate", "channels"])

def get_info(path):
    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        # new version of torchaudio
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)


def find_audio_files(path, progress=True, n_samples_limit=-1):
    with open(path) as f:
        audio_files = []
        for root, folders, files in os.walk(path, followlinks=True):
            for file in files:
                file = Path(root) / file
                if file.suffix.lower() in ['wav']:
                    audio_files.append(str(file.resolve()))
        meta = []
        if n_samples_limit > 0:
            audio_files = audio_files[:n_samples_limit]
        for idx, file in enumerate(audio_files):
            info = get_info(file)
            meta.append((file, info.length))
            if progress:
                print(format((1 + idx) / len(audio_files), " 3.1%"), end='\r', file=sys.stderr)
        meta.sort()
        return meta



"""
usage:

out=egs/mydataset/tr
mkdir -p $out
python -m src.prep_egs_files $lr > $out/lr.json
python -m src.prep_egs_files $hr > $out/hr.json

"""
if __name__ == "__main__":
    meta = []
    path = sys.argv[1]
    n_samples_limit = int(sys.argv[2])
    meta += find_audio_files(path, n_samples_limit=n_samples_limit)
    json.dump(meta, sys.stdout, indent=4)