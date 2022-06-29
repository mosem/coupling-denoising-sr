# Coupling Audio Denoising and Super-Resolution

## Steps

### Install required libraries

The code requires the following libraries:
- pytorch
- torchaudio
- hydra
- sox
- numpy
- tqdm
- openunmix
- pesq
- pystoi
- wandb
- cv2
- **complete**... add requirements.txt file

It was tested on Cuda/11.3. When installing pytorch, make sure to download the relevant version:
`pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 --no-cache-dir`

### Download and Prepare Data

We use the Valentini (56 speakers) dataset. Download and extract the following files
from [here](https://datashare.ed.ac.uk/handle/10283/2791):
- clean_trainset_56spk_wav.zip
- noisy_trainset_56spk_wav.zip
- clean_testset_wav.zip
- noisy_testset_wav.zip

Our code runs on 3 different up-sampling ratios settings: 8kHz -> 16kHz, 4kHz -> 16kHz, 8kHz -> 24kHz.
Therefore we need to down-sample the data from 48kHz to the 4 sample rates (4,8,16,24), and upsample the source 
sample-rates (4,8) to the target sample-rates (4->16, 8->16, 8->24).

To prepare the data, run: `bash prep_valentini_data <directory of valentini dataset>`

This resamples the data in multi-threaded fashion.

To comply with existing code, place the datasets in `data/valentini` where `data` is a sibling directory to your
root script folder (e.g. `coupling-denoising-sr`). Otherwise, change the `data_root_dir` parameter in `
coupling-denoising-sr/data/prep_egs_files.sh` to point to the relevant directory.

### Create egs Files

Our code requires files that list the names of audio files.
To create these files, run: `bash create_egs_files`

This creates all the relevant files.

### Run Script Files

All script files are placed under the bash_scripts folder.
Each script file runs the train and test scripts in sequence.

We use the Hydra framework to configure the code.
For example, to run on multiple GPUs, modify `distributed` to `True` in `diffusion/conf/main_config.yaml`.