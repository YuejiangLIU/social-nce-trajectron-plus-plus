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
bash run_train.sh ${DATASET} 0.0 && bash run_eval.sh ${DATASET} 0.0
```

#### Contrastive ####

To train the Trajectron++ with Social-NCE, run the following command:
```bash
bash run_train.sh ${DATASET} && bash run_eval.sh ${DATASET}
```

#### Comparison ####

To compare different models, run the following command:
```bash
python benchmark.py --dataset ${DATASET}
```

### Basic Results ###

The scripts above yield the following results (on GeForce RTX 3090). The result may subject to mild variance on different GPU devices. More details will be released soon!

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
    <td align="center">80</td>
    <td align="center">0.215</td>
    <td align="center">0.461</td>
    <td align="center">4.942</td>
    <td align="center">0.211</td>
    <td align="center">0.460</td>
    <td align="center">3.605</td>
  </tr>
  <tr>
    <td align="center">90</td>
    <td align="center">0.208</td>
    <td align="center">0.453</td>
    <td align="center">4.527</td>
    <td align="center">0.207</td>
    <td align="center">0.455</td>
    <td align="center">3.534</td>
  </tr>
  <tr>
    <td align="center">100</td>
    <td align="center">0.208</td>
    <td align="center">0.454</td>
    <td align="center">4.149</td>
    <td align="center">0.204</td>
    <td align="center">0.451</td>
    <td align="center">3.293</td>
  </tr>
  <tr>
    <td align="center">Average</td>
    <td align="center">0.210</td>
    <td align="center">0.456</td>
    <td align="center">4.539</td>
    <td align="center">0.207</td>
    <td align="center">0.455</td>
    <td align="center">3.477</td>
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