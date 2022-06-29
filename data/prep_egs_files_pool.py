# This code is from FAIR's Demucs denoiser: https://github.com/facebookresearch/denoiser

from pathlib import Path
import os
import torchaudio
from collections import namedtuple
import json
import sys
from multiprocessing import Pool, Manager
import tqdm

N_PROCESSES = 10

Info = namedtuple("Info", ["length", "sample_rate", "channels"])
manager = Manager()
meta_list = manager.list()


def get_info(path):
    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        # new version of torchaudio
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)


def appendInfoToMetaList(file):
    global meta_list
    info = get_info(file)
    meta_list.append((file, info.length))


def find_audio_files(path, n_samples_limit=-1, progress=True):
    audio_files = []
    for file in os.listdir(path):
        if file.endswith('.wav'):
            audio_files.append(os.path.join(path, file))

    if n_samples_limit > 0:
        audio_files = audio_files[:n_samples_limit]

    pool = Pool(processes=N_PROCESSES)
    for _ in tqdm.tqdm(pool.imap(appendInfoToMetaList, audio_files), total=len(audio_files)):
        pass

"""
usage:

out=egs/mydataset/tr
mkdir -p $out
python -m src.prep_egs_files $lr > $out/lr.json
python -m src.prep_egs_files $hr > $out/hr.json

"""

if __name__ == "__main__":
    path = sys.argv[1]
    n_samples_limit = int(sys.argv[2])
    find_audio_files(path, n_samples_limit=n_samples_limit)
    json.dump(sorted(list(meta_list)), sys.stdout, indent=4)