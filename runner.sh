#!/bin/bash
#SBATCH --time=0-02:30:00
#SBATCH --account=def-npopli
#SBATCH --mem=32000M            # memory per node
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10      # CPU cores/threads


source ./venv/bin/activate
pip install --no-index --upgrade pip
pip install -r ./requirements.txt
python ./adversial_and_physical_attacks.py
deactivate

