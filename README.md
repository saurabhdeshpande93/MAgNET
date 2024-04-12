## MAgNET: A Graph U-Net Architecture for Mesh-Based Simulations  

MAgNET is a graph U-Net architecture composed of two novel deep learning layers. The first layer is the Multichannel Aggregation (MAg) layer, which expands upon the idea of multichannel localized operations in convolutional neural networks to accommodate arbitrary graph-structured inputs. And second type of layers are pooling/unpooling layers, designed to facilitate efficient learning on high-dimensional inputs by utilizing reduced representations of arbitrary graph-structured inputs.

<br />

![schematic](graphunet.jpg)

<br />

Sources for MAg and GPool/GUnpool are located in [<span style="color:blue">layers</span>](src/main/layers).


## Implementation to non-linear Finite Element dataset

We demonstrate the predictive capabilities of MAgNET in surrogate modeling for non-linear finite element simulations in the mechanics of solids. The supplementary finite element simulation data utilised in the paper is available on [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7784804.svg)](https://doi.org/10.5281/zenodo.7784804).

While the scripts to generate non-linear FEM datasets are available in the [<span style="color:blue">femscripts</span>](src/femscripts) directory.


<br />

## Dependencies

Scripts have been tested running under Python 3.9.5, with the following packages installed (along with their dependencies). In addition, CUDA 10.1 and cuDNN 7 have been used.


- `tensorflow-gpu==2.4.1`
- `keras==2.4.3`
- `numpy==1.19.5`
- `pandas==1.3.4`
- `scikit-learn==1.0.1`

All the finite element simulations are performed using the [AceFEM](http://symech.fgg.uni-lj.si/Download.htm) library.

<br />

## Instructions

1. Download the supplementary data from zenodo and keep it in the src directory.

2. MAgNET architectures can be used for inference using the pre-trained weights or can be trained from scratch by running `main.py` script in the [<span style="color:blue">visualisation</span>](src/main) directory.

3. Use `post.py` to save the example of interest to further visualise it in Acegen using Mathematica notebooks present in the [<span style="color:blue">visualisation</span>](src/postprocess/visualisation) directory.

<br />

## Cite

Cite our paper if you use this code in your own work:

```
@article{DESHPANDE2024108055,
title = {MAgNET: A graph U-Net architecture for mesh-based simulations},
journal = {Engineering Applications of Artificial Intelligence},
volume = {133},
pages = {108055},
year = {2024},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2024.108055},
url = {https://www.sciencedirect.com/science/article/pii/S0952197624002136},
author = {Saurabh Deshpande and Stéphane P.A. Bordas and Jakub Lengiewicz},
keywords = {Geometric deep learning, Mesh based simulations, Finite element method, Graph U-Net, Surrogate modeling}
}
```

<br />
