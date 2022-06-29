#!/bin/bash

# dummy file

echo "creating egs files"

bash data/prep_egs_files.sh noisy 8 clean 16 4

# SR-only on noisy files

bash data/prep_egs_files.sh noisy 4 noisy 16
bash data/prep_egs_files.sh noisy 8 noisy 16
bash data/prep_egs_files.sh noisy 8 noisy 24

# SR-only on clean files

bash data/prep_egs_files.sh clean 4 clean 16
bash data/prep_egs_files.sh clean 8 clean 16
bash data/prep_egs_files.sh clean 8 clean 24

# Denoise only

bash data/prep_egs_files.sh noisy 4 clean 4
bash data/prep_egs_files.sh noisy 8 clean 8

# Denoise + SR

bash data/prep_egs_files.sh noisy 4 clean 16
bash data/prep_egs_files.sh noisy 8 clean 16
bash data/prep_egs_files.sh noisy 8 clean 24

echo "done creating egs files"