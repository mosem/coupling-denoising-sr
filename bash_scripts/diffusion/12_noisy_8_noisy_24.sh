#!/bin/bash

python diffusion/train.py dset=noisy-8-noisy-24

python diffusion/test.py dset=noisy-8-noisy-24
