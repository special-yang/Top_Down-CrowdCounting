# CrowdFormer: An Overlap Patching Vision Transformer for Top-Down Crowd Counting（IJCAI2022）
Abstract:Recent crowd counting methods typically predict a density map as an intermediate representation of counting, and achieve good performance. However, due to the perspective phenomenon, there is a scale variation in real scenes, which causes current density map-based methods suffer from a severe scene generalization problem because only a limited number of scales are fitted in their density map prediction and generation. To address this issue, we propose a novel vision transformer network, i.e.,CrowdFormer, and an adaptive framework of density kernels fusion for more accurate density map estimation and generation, respectively. Thereafter, we incorporate these two innovations into an adaptive crowd counting model, which takes both the annotation dot map and original image as input, and jointly learns the density map estimator and generator within an end-to-end framework. The experimental results demonstrate that the proposed model achieves the state-of-the-art in the terms of MAE
and MSE (e.g., it achieved a MAE of 67.1 and MSE of 301.6 on NWPU-Crowd dataset.), and confirm the effectiveness of the proposed two designs.

## CrowdFormer network

## Results
![results.png] (results.png)
