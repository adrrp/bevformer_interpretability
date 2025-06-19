## Abstract

This repository extends the cutting-edge **BEVFormer** framework, a novel approach for learning unified Bird's-Eye-View (BEV) representations using spatiotemporal transformers to support various autonomous driving perception tasks. The original BEVFormer work, as described by its authors, effectively exploits both spatial and temporal information by interacting with spatial and temporal space through predefined grid-shaped BEV queries. They achieved state-of-the-art results with **56.9% NDS** on the nuScenes test set.

A significant portion of the core BEVFormer architecture and the initial BEV segmentation capabilities within this repository are derived from the work found in the **[BEVFormer_segmentation_detection](https://github.com/Bin-ze/BEVFormer_segmentation_detection/tree/master)** repository.All the attributions to the original authors for their foundational contribution.

Building upon this robust foundation, my contribution focuses on integrating **interpretability scripts** to provide deeper insights into the model's decision-making process. These interpretability scripts are my own addition, aiming to enhance the understanding and analysis of this powerful model.


## Usage

For the proper installation the docker file can be used along with the instructions to prepare the dataset in /docs/. For the models it's necessary to use the segmentation and detection models from BEVFormer_segmentation_detection  **[BEVFormer_segmentation_detection](https://github.com/Bin-ze/BEVFormer_segmentation_detection/tree/master)** repository (the exact version for the experiments on my thesis has been the bevformer_base_seg_det_150 version.

When the repo is properly installed, the following code releated to my Bachelor's Thesis can be found:
-In tools/test.py an extended version of the original test.py is implemented allowing the user to capture the intermediate BEV features into .npy maps and then use the bev_features/bev_hook.py script to visualize them
-In the saliency_techniques folder are the necessary tools for generating, evaluating and replicating the experimentation results can be found. Concretly:
    -standard_saliency.py can generate all the types of saliency used in my Thesis
    -perturbation_test.py performs perturbation tests
    -generate_perturbed_data.py contains the necessary function to generate an specific perturbation test.
    -metrics.py takes the results from perturbation tests and generates their AUC plots.



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

These tools provide a framework for both visual and quantitative analysis of the model’s behavior, helping to better understand and evaluate the interpretability of BEV-based perception systems.
