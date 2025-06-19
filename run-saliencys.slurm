#!/bin/bash
#SBATCH --partition=gpi.compute
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,gpumem:32G
#SBATCH --time=8:00:00

#PYTHONPATH=$(pwd) python ./saliency_techniques/perturbation_test.py ./projects/configs/bevformer/bevformer_base_seg_det_150x150.py ./ckpts/bevformer_base_seg_det_150.pth

PYTHONPATH=$(pwd) python ./saliency_techniques/standard_saliency.py ./projects/configs/bevformer/bevformer_base_seg_det_150x150.py ./ckpts/bevformer_base_seg_det_150.pth --eval segms
#  --show \
#  --show-dir ./outputs/model
