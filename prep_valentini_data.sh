#!/bin/bash

valentini_dir_path=$1

# clean train sets

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/clean_trainset_56spk_wav \
	--out_dir ${valentini_dir_path}/clean_trainset_56spk_24k \
	--target_sr 24000 \

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/clean_trainset_56spk_wav \
	--out_dir ${valentini_dir_path}/clean_trainset_56spk_16k \
	--target_sr 16000 \

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/clean_trainset_56spk_wav \
	--out_dir ${valentini_dir_path}/clean_trainset_56spk_8k \
	--target_sr 8000 \

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/clean_trainset_56spk_wav \
	--out_dir ${valentini_dir_path}/clean_trainset_56spk_4k \
	--target_sr 4000 \

# clean train sets from low sample rate to target sample rate: 8->24, 8->16, 4->16

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/clean_trainset_56spk_8k \
	--out_dir ${valentini_dir_path}/clean_trainset_56spk_8_24k \
	--target_sr 24000 \

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/clean_trainset_56spk_8k \
	--out_dir ${valentini_dir_path}/clean_trainset_56spk_8_16k \
	--target_sr 16000 \

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/clean_trainset_56spk_4k \
	--out_dir ${valentini_dir_path}/clean_trainset_56spk_4_16k \
	--target_sr 16000 \

# noisy train sets

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/noisy_trainset_56spk_wav \
	--out_dir ${valentini_dir_path}/noisy_trainset_56spk_24k \
	--target_sr 24000 \

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/noisy_trainset_56spk_wav \
	--out_dir ${valentini_dir_path}/noisy_trainset_56spk_16k \
	--target_sr 16000 \

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/noisy_trainset_56spk_wav \
	--out_dir ${valentini_dir_path}/noisy_trainset_56spk_8k \
	--target_sr 8000 \

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/noisy_trainset_56spk_wav \
	--out_dir ${valentini_dir_path}/noisy_trainset_56spk_4k \
	--target_sr 4000 \

# noisy train sets from low sample rate to target sample rate: 8->24, 8->16, 4->16

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/noisy_trainset_56spk_8k \
	--out_dir ${valentini_dir_path}/noisy_trainset_56spk_8_24k \
	--target_sr 24000 \

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/noisy_trainset_56spk_8k \
	--out_dir ${valentini_dir_path}/noisy_trainset_56spk_8_16k \
	--target_sr 16000 \

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/noisy_trainset_56spk_4k \
	--out_dir ${valentini_dir_path}/noisy_trainset_56spk_4_16k \
	--target_sr 16000 \


# noisy test sets

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/noisy_testset_wav \
	--out_dir ${valentini_dir_path}/noisy_testset_24k \
	--target_sr 24000 \

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/noisy_testset_wav \
	--out_dir ${valentini_dir_path}/noisy_testset_16k \
	--target_sr 16000 \

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/noisy_testset_wav \
	--out_dir ${valentini_dir_path}/noisy_testset_8k \
	--target_sr 8000 \

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/noisy_testset_wav \
	--out_dir ${valentini_dir_path}/noisy_testset_4k \
	--target_sr 4000 \

# clean test sets from low sample rate to target sample rate: 8->24, 8->16, 4->16

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/noisy_testset_8k \
	--out_dir ${valentini_dir_path}/noisy_testset_8_24k \
	--target_sr 24000 \

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/noisy_testset_8k \
	--out_dir ${valentini_dir_path}/noisy_testset_8_16k \
	--target_sr 16000 \

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/noisy_testset_4k \
	--out_dir ${valentini_dir_path}/noisy_testset_4_16k \
	--target_sr 16000 \

# clean test sets

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/clean_testset_wav \
	--out_dir ${valentini_dir_path}/clean_testset_24k \
	--target_sr 24000 \

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/clean_testset_wav \
	--out_dir ${valentini_dir_path}/clean_testset_16k \
	--target_sr 16000 \

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/clean_testset_wav \
	--out_dir ${valentini_dir_path}/clean_testset_8k \
	--target_sr 8000 \

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/clean_testset_wav \
	--out_dir ${valentini_dir_path}/clean_testset_4k \
	--target_sr 4000 \

# clean test sets from low sample rate to target sample rate: 8->24, 8->16, 4->16

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/clean_testset_8k \
	--out_dir ${valentini_dir_path}/clean_testset_8_24k \
	--target_sr 24000 \

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/clean_testset_8k \
	--out_dir ${valentini_dir_path}/clean_testset_8_16k \
	--target_sr 16000 \

python3 data/prep_audio_data.py  \
	--data_dir ${valentini_dir_path}/clean_testset_4k \
	--out_dir ${valentini_dir_path}/clean_testset_4_16k \
	--target_sr 16000 \