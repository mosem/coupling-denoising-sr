#!/bin/bash

python diffusion/train.py dset=noisy-4-noisy-16

python diffusion/test.py dset=noisy-4-noisy-16
