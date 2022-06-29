#!/bin/bash

python diffusion/train.py dset=clean-8-noisy-16

python diffusion/test.py dset=clean-8-noisy-16