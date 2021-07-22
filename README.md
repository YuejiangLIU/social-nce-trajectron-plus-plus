## Trajectron++ with Social-NCE

<p align="center">
  <img src="docs/illustration.png" width="300">
</p>

This is an official implementation of the Social-NCE applied to the Trajectron++ forecasting model.

**[Social NCE: Contrastive Learning of Socially-aware Motion Representations](https://arxiv.org/abs/2012.11717)**
<br>by
<a href="https://sites.google.com/view/yuejiangliu/">Yuejiang Liu</a>,
<a href="https://qiyan98.github.io/">Qi Yan</a>,
<a href="https://people.epfl.ch/alexandre.alahi/?lang=en/">Alexandre Alahi</a> at
<a href="https://www.epfl.ch/labs/vita/">EPFL</a>
<br>
to appear at ICCV 2021

TL;DR: Contrastive Representation Learning + Negative Data Augmentations &#129138; Robust Neural Motion Models
> * Rank in 1st place on the [Trajnet++ challenge](https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge/leaderboards) since November 2020 to present
> * Significantly reduce the collision rate of SOTA [human trajectroy forecasting models](https://github.com/StanfordASL/Trajectron-plus-plus)
> * SOTA on imitation / reinforcement learning for [autonomous navigation in crowds](https://github.com/vita-epfl/CrowdNav)

Please check out our code for experiments on different models as follows:  
**[Social NCE + Trajectron](https://github.com/YuejiangLIU/social-nce-trajectron-plus-plus)  |  [Social NCE + STGCNN](https://github.com/qiyan98/social-nce-stgcnn)  |  [Social NCE + CrowdNav](https://github.com/vita-epfl/social-nce-crowdnav)**

### Preparation

Setup environments follwoing the [SETUP.md](docs/SETUP.md)

### Training & Evaluation ###

To train a model on the ETH / UCY Pedestrian datasets, the dataset needs to be specified, e.g.,
```bash
DATASET=univ
```

#### Baseline ####

The vanilla Trajectron++ model can be trained and evaluated as follows: 
```bash
bash scripts/run_train.sh ${DATASET} 0.0 && bash scripts/run_eval.sh ${DATASET} 0.0
```

#### Contrastive ####

To train the Trajectron++ with Social-NCE, run the following command:
```bash
bash scripts/run_train.sh ${DATASET} && bash bash scripts/run_eval.sh ${DATASET}
```

#### Comparison ####

To search for hyper-parameters on different datasets, run the following [bash scripts](scripts):
```bash
bash scripts/run_<dataset>.sh
```

Our [pre-trained models](https://drive.google.com/file/d/1APAIlgJS9BDZHFCvwMrfzj9z_9DSS6LB/view?usp=sharing) can be downloaded as follows:
```bash
gdown https://drive.google.com/uc?id=1APAIlgJS9BDZHFCvwMrfzj9z_9DSS6LB
unzip pretrained_trajectron++.zip -d experiments/pedestrians/models
```

To compare different models, run the following command:
```bash
python benchmark.py --dataset ${DATASET}
```

### Basic Results ###

The scripts above yield the following results (on GeForce RTX 3090). The result may subject to [mild variance](https://github.com/StanfordASL/Trajectron-plus-plus/issues/38#issuecomment-810612481) on different GPU devices.

On average, our method reduces the collision rate of the Trajectron++ by over 45%, without degrading its performance in terms of prediction accuracy and diversity.

<table>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<thead>
  <tr>
    <th rowspan="2">Epoch</th>
    <th colspan="3">Trajectron++ w/o Ours</th>
    <th colspan="3">Trajectron++ w/ Ours</th>
  </tr>
  <tr>
    <td align="center">ADE</td>
    <td align="center">FDE</td>
    <td align="center">COL</td>
    <td align="center">ADE</td>
    <td align="center">FDE</td>
    <td align="center">COL</td>
  </tr>
</thead>
<!-- TABLE BODY -->
<tbody>
  <tr>
    <td align="center">ETH</td>
    <td align="center">0.388</td>
    <td align="center">0.810</td>
    <td align="center">1.156</td>
    <td align="center">0.386</td>
    <td align="center">0.791</td>
    <td align="center">0.000</td>
  </tr>
  <tr>
    <td align="center">HOTEL</td>
    <td align="center">0.110</td>
    <td align="center">0.184</td>
    <td align="center">0.837</td>
    <td align="center">0.107</td>
    <td align="center">0.177</td>
    <td align="center">0.381</td>
  </tr>
  <tr>
    <td align="center">UNIV</td>
    <td align="center">0.199</td>
    <td align="center">0.450</td>
    <td align="center">3.378</td>
    <td align="center">0.195</td>
    <td align="center">0.435</td>
    <td align="center">3.079</td>
  </tr>
  <tr>
    <td align="center">ZARA1</td>
    <td align="center">0.148</td>
    <td align="center">0.320</td>
    <td align="center">0.462</td>
    <td align="center">0.150</td>
    <td align="center">0.330</td>
    <td align="center">0.178</td>
  </tr>
  <tr>
    <td align="center">ZARA2</td>
    <td align="center">0.114</td>
    <td align="center">0.250</td>
    <td align="center">1.027</td>
    <td align="center">0.114</td>
    <td align="center">0.255</td>
    <td align="center">0.993</td>
  </tr>
  <tr>
    <td align="center">Average</td>
    <td align="center">0.192</td>
    <td align="center">0.403</td>
    <td align="center">1.372</td>
    <td align="center">0.191</td>
    <td align="center">0.398</td>
    <td align="center">0.926</td>
  </tr>
</tbody>
</table>

### Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{liu2020snce,
  title   = {Social NCE: Contrastive Learning of Socially-aware Motion Representations},
  author  = {Yuejiang Liu and Qi Yan and Alexandre Alahi},
  journal = {arXiv preprint arXiv:2012.11717},
  year    = {2020}
}
```

### Acknowledgement

Our code is developed upon the official implementation of [Trajectron++](https://github.com/StanfordASL/Trajectron-plus-plus).
