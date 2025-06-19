#!/bin/bash
#SBATCH --partition=gpi.compute
#SBATCH --mem=12G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,gpumem:16G
#SBATCH --time=6:00:00

PYTHONPATH=$(pwd) python ./saliency_techniques/perturbation_test.py ./projects/configs/bevformer/bevformer_base_seg_det_150x150.py ./ckpts/bevformer_base_seg_det_150.pth
#  --show \
#  --show-dir ./outputs/model
