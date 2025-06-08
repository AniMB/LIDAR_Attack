#!/bin/bash


nvidia-smi

module load python/3.11

virtualenv --no-download ./venv

sbatch ./runner.sh