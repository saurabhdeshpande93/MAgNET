## MAgNET: A Graph U-Net Architecture for Mesh-Based Simulations  

This repository provides the implementations of "[MAgNET: A Graph U-Net Architecture for Mesh-Based Simulations](https://arxiv.org/abs/2211.00713)".

Supplementary data for the paper is available on [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7784804.svg)](https://doi.org/10.5281/zenodo.7784804).



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

## Cite

Cite our paper if you use this code in your own work:

```
@misc{deshpande2023magnet,
      title={MAgNET: A Graph U-Net Architecture for Mesh-Based Simulations},
      author={Saurabh Deshpande and Stéphane P. A. Bordas and Jakub Lengiewicz},
      year={2023},
      doi = {https://doi.org/10.48550/arXiv.2211.00713},
      url = {https://arxiv.org/abs/2211.00713},
      eprint={2211.00713},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

<br />
