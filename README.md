## Abstract

This repository builds upon the **BEVFormer** framework, a novel approach for learning unified Bird’s-Eye-View (BEV) representations through spatiotemporal transformers, enabling a wide range of autonomous driving perception tasks. BEVFormer leverages spatial and temporal information via predefined grid-shaped BEV queries, achieving **56.9% NDS** on the nuScenes test set.

The base architecture and initial BEV segmentation capabilities used here are adapted from the [BEVFormer_segmentation_detection](https://github.com/Bin-ze/BEVFormer_segmentation_detection/tree/master) repository.

This extended version introduces **interpretability tools** specifically designed to gain insights into the model’s internal decision-making processes. These contributions include a suite of saliency-based visualization and evaluation techniques developed as part of my Bachelor's Thesis.

## Usage

To get started, follow the installation instructions provided in the `/docs/` directory. A Dockerfile is included to ensure a consistent environment setup.

### Model Requirements

Download and use the **segmentation and detection models** from the [BEVFormer_segmentation_detection](https://github.com/Bin-ze/BEVFormer_segmentation_detection/tree/master) repository. The specific version used in this project is:  
`bevformer_base_seg_det_150`.

### Thesis-related Code

After setting up the repository, you will find the following tools related to the interpretability experiments:

- `tools/test.py`:  
  Extended from the original BEVFormer version, this script allows users to evaluate the model and also to extract intermediate BEV feature maps and save them as `.npy` files. These features can be visualized using:

- `bev_features/bev_hook.py`:  
  Script for visualizing intermediate BEV features captured during inference.

- `saliency_techniques/`:  
  Folder containing the core interpretability tools developed for this thesis:
  - `standard_saliency.py`: Generates various saliency maps used in the experiments and potentially visualize them.
  - `perturbation_test.py`: Performs robustness tests by perturbing inputs.
  - `generate_perturbed_data.py`: Helper functions to create perturbed samples for testing.
  - `metrics.py`: Computes and plots AUC metrics from the perturbation experiments.


