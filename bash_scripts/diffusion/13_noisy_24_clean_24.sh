#!/bin/bash

python diffusion/train.py dset=noisy-24-clean-24

python diffusion/test.py dset=noisy-24-clean-24
