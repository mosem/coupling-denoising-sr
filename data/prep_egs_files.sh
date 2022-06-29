#!/bin/bash

# usage example: bash prep_egs_files.sh noisy 8 clean 16

cd ./data

source_mode=$1
source_sr=$2
target_mode=$3
target_sr=$4
n_samples_limit=${5:--1}

N_SPEAKERS=56 # 56/28

out_dir=$source_mode-$source_sr-$target_mode-$target_sr

data_root_dir=$(realpath ../../data/valentini)

echo "assuming data is in ${data_root_dir}"

if [[ $n_samples_limit -gt 0 ]]
then
  out_dir+="(${n_samples_limit})"
fi

echo "saving to valentini/${out_dir}"

tr_out=../egs/valentini/$out_dir/tr
val_out=../egs/valentini/$out_dir/val

source_train_dir=${data_root_dir}/${source_mode}_trainset_${N_SPEAKERS}spk_${source_sr}k
target_train_dir=${data_root_dir}/${target_mode}_trainset_${N_SPEAKERS}spk_${target_sr}k

source_val_dir=${data_root_dir}/${source_mode}_testset_${source_sr}k
target_val_dir=${data_root_dir}/${target_mode}_testset_${target_sr}k

mkdir -p $tr_out
mkdir -p $val_out

echo "preparing egs files for train files"
python -m prep_egs_files $source_train_dir $n_samples_limit > $tr_out/source.json
python -m prep_egs_files $target_train_dir $n_samples_limit > $tr_out/target.json

echo "preparing egs files for validation files"
python -m prep_egs_files $source_val_dir $n_samples_limit > $val_out/source.json
python -m prep_egs_files $target_val_dir $n_samples_limit > $val_out/target.json

if [ "$source_sr" != "$target_sr" ]; then
    echo "preparing egs files for upsampled files (train and validation): ${source_sr}->${target_sr}"
    upsampled_source_dir=${data_root_dir}/${source_mode}_trainset_${N_SPEAKERS}spk_${source_sr}_${target_sr}k
    upsampled_val_dir=${data_root_dir}/${source_mode}_testset_${source_sr}_${target_sr}k
    python -m prep_egs_files $upsampled_source_dir $n_samples_limit > $tr_out/source_up.json
    python -m prep_egs_files $upsampled_val_dir $n_samples_limit > $val_out/source_up.json
fi