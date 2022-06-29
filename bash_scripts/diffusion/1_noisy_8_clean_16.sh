#!/bin/bash

python diffusion/train.py dset=noisy-8-clean-16

python diffusion/test.py dset=noisy-8-clean-16
