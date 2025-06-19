#!/bin/bash
#SBATCH --partition=gpi.compute
#SBATCH --mem=8G
#SBATCH --gres=gpu:1,gpumem:8G
PYTHONPATH=$(pwd) python ./visual_det_seg.py 
