# Overview

This repository contains the implementation of the KFBK algorithm described in our paper:  
"A High-Precision CSI-Based Localization Framework with Kolmogorov-Arnold Network and Broad Learning System".

KFBK leverages wavelet-based denoising, Kolmogorov-Arnold Network (KAN), and Broad Learning System (BLS) with a dynamic fusion mechanism to achieve efficient and accurate indoor localization using CSI data.



# Abstract

Fingerprint-based indoor localization has become a crucial technology due to its wide availability, low hardware costs, and increasing demand for location-based services. However, existing methods face challenges such as:

- Sensitivity to noise and multipath interference in CSI signals  
- Inefficient feature extraction from high-dimensional data  
- High computational complexity of deep learning models  

KFBK addresses these challenges with an effective localization framework that integrates signal preprocessing, nonlinear feature learning, and adaptive model fusion.

Key features of KFBK include:



## Offline Training Stage

- CSI data are preprocessed using multi-level wavelet decomposition with adaptive thresholding to suppress noise and enhance signal quality.  
- A Kolmogorov-Arnold Network (KAN) is employed to extract nonlinear features and reduce data dimensionality while preserving essential information.  
- A Broad Learning System (BLS) is trained using the extracted features to build an efficient localization model.  



## Online Localization Stage

- Real-time CSI data are processed using the trained KAN and BLS models.  
- A temperature-controlled dynamic weighting mechanism adaptively fuses the outputs of KAN and BLS.  
- The final location is estimated based on the weighted combination of both models.  



Experimental results demonstrate that KFBK outperforms several state-of-the-art algorithms in real-world indoor environments, achieving higher localization accuracy and robustness while maintaining computational efficiency.

## Usage
The main scripts in this repository include:

- Main-lab.py: Responsible for the entire localization workflow in the lab scenario, including the construction and training of the end-to-end localization model.

- Main-meeting.py: This script is used for the meeting room scenario.

## Citation

If you use this work for your research, please cite:

```bibtex
@INPROCEEDINGS{11323022,
  author={He, Xuanqi and Zhang, Mingbo and Zhu, Xiaoqiang and Yao, Yingying and Wang, Chenyang and Li, Lingkun},
  booktitle={2025 IEEE 31th International Conference on Parallel and Distributed Systems (ICPADS)}, 
  title={A High-Precision CSI-Based Localization Framework with Kolmogorov-Arnold Network and Broad Learning System}, 
  year={2025},
  volume={},
  number={},
  pages={1-8},
  keywords={Location awareness;Learning systems;Accuracy;Noise reduction;Noise;Fingerprint recognition;Feature extraction;Integrated sensing and communication;Nonlinear dynamical systems;Splines (mathematics);ISAC;Fingerprint Localization;CSI;KAN;BLS},
  doi={10.1109/ICPADS67057.2025.11323022}
}
