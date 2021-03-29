### Environment Setup ###

Note: require python 3.6
```bash
conda create --name trajectron++ python=3.6 -y
source activate trajectron++
pip install -r requirements.txt
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

### Data Setup ###

#### Pedestrian Datasets ####

The preprocessed data splits for the ETH and UCY Pedestrian datasets can be found in the [original public repository of Trajectron++](https://github.com/StanfordASL/Trajectron-plus-plus/tree/master/experiments/pedestrians/raw). Please download these data and place them in the same location of this repository. 

In order to process them into a data format that our model can work with, execute the follwing.
```bash
cd experiments/pedestrians
python process_data.py  # This will take around 10-15 minutes, depending on your computer.
```
