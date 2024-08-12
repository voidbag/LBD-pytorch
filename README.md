# LBD-pytorch
## Lightweight Method for Modeling Confidence in Recommendations with Learned Beta Distributions, RecSys2023
The 1st implementation of Learned Beta Distribution Model using torch.special.betainc of PyTorch (RecSys2023).\
The original version LBD (https://github.com/NKNY/confidencerecsys2023) was implemented in Tensorflow.\
When I tried to implement LBD model in PyTorch, betainc(regularized incomplete beta function) implementation didn't exist in PyTorch.\
To use LBD model in PyTorch, I must have implemented the betainc first, referring to tensorflow-probability source code.\
torch.special.betainc I implemented is faster than the version of tensorflow-probability, because it is written in c++, not python.


The implementation of 'torch.special.betainc' is under reveiw (https://github.com/pytorch/pytorch/pull/132135). \
You should build the source in the PR or download docker image for running this code.

## Installation
### Pull Docker Image
`docker pull voidbag/pytorch:2.4.0-cuda12.1-cudnn9-devel-wip-betainc-with-bwd-voidbag-v0.1.0`

### Install Requirements
`pip install -r ./requirements.txt`

## Run
### Preprocessing
`python ./download_n_preprocess_ml_10m.py`
### Run Baseline
Configuration in baseline in origin repo \
`python ./main.py --model OrdRec --model-config-json ./OrdRec-UI_512.json --lr 7.227590331297689e-05` \
If you don't want 10-fold cross-validation, set cross-validation as 'false' like this. \
`python ./main.py --model OrdRec --model-config-json ./OrdRec-UI_512.json --lr 7.227590331297689e-05 --no-cross-validation`
### Run LBD model
Configuration in LBD in origin repo \
`python ./main.py --model LBD --model-config-json ./LBDA_512_sum_ab.json --lr 0.001` \
If you don't want 10-fold cross-validation, set cross-validation as 'false' like this. \
`python ./main.py --model LBD --model-config-json ./LBDA_512_sum_ab.json --lr 0.001 --no-cross-validation`
### Get Result
`python ./analyze_result.py --dir-lbd ./output/LBDA_512_sum_ab/ --dir-ordrec ./output/OrdRec-UI_512/ --out-prefix ./result_plot`

#### Accuracy
The performance of OrdRec-UI has been improved with full-length random sampling, not using buffer 10000
|           | accuracy          |
|:----------|:------------------|
| OrdRec-UI | 0.43097(0.000206) |
| LBD-A     | 0.43500(0.000128) |

#### Correlation
|                                  | OrdRec-UI        | LBD-A            |
|:---------------------------------|:-----------------|:-----------------|
| Pearson's r (Linear Correlation) | 0.25396(0.00022) | 0.33407(0.00028) |
| Kendall’s t (Rank   Correlation) | 0.16179(0.00017) | 0.21901(0.00023) |

#### Scatter(mae, Predicted Variance(Normalized_quantile))
![plot_predicted_variance(normalized)_mae](https://github.com/user-attachments/assets/c083ea6c-5116-499d-976a-b2c77e81a582)
#### Precision@1
![precision_1_plot](https://github.com/user-attachments/assets/2db01380-564c-4034-821b-fbbec2c2ea62)

