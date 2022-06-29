#!/bin/bash

python diffusion/train.py dset=clean-4-clean-16

python diffusion/test.py dset=clean-4-clean-16
