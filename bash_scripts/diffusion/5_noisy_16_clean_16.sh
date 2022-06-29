#!/bin/bash

python diffusion/train.py dset=noisy-16-clean-16

python diffusion/test.py dset=noisy-16-clean-16