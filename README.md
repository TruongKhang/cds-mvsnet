# [ICLR2022] CDS-MVSNet
This is an official implementation of our paper [Curvature-guided dynamic scale networks for multi-view stereo](https://arxiv.org/pdf/2112.05999.pdf), which was accepted at ICLR 2022.

Our method can perform reconstruction in real-time due to fast run-time and low-memory consumption. Here is a demo with the input video captured by a smartphone. We applied a SLAM method for camera pose estimation and then used CDS-MVSNet for dense reconstruction.

![Alt Text](demo.gif)

## TODO List
- [x] Make repo clean
- [x] Upload [all pretrained models](pretrained/), [DTU](https://kaist.gov-dooray.com/share/drive-files/kkmh2qu2awzo.er70beOnTO64mUhhGH81_A) and [TanksandTemples](https://kaist.gov-dooray.com/share/drive-files/kkmh2qu2awzo.ptZ1DiyZRuC6NaHSAs3a0A) pointclouds.

## Requirements
    conda create -n venv python=3.6
    conda activate venv
    conda install pytorch==1.6.0 torchvision cudatoolkit=10.1 -c pytorch
    pip install -r requirements.txt

## Training
The training datasets including DTU and BlendedMVS are provided by [Yao Yao](https://github.com/YoYo000/MVSNet). 
 All training parameters are configured in a configuration file `*.json`. We provide two options for training

### Training on DTU and then fine-tuning on BlendedMVS
To train model on DTU dataset, change the directory of DTU dataset in the configuration file. Then, run this command:

    python train.py --config configs/config_dtu.json
    
After the training is finished, the train model will be saved in `saved/models/CDS-MVSNet/<date_and_year>`. 
To fine-tune on BlendedMVS dataset, you need to update the file `config.json` in `saved/models/CDS-MVSNet/<date_and_year>/config.json` by contents in file `config_blended.json`.
Then, run this command to start fine-tuning on BlendedMVS

    python train.py --resume saved/models/CDS-MVSNet/<date_and_year>/checkpoint-epoch30.pth
    
The model will be trained continued.

### Training on DTU and BlendedMVS simultaneously.
 
Update the directory of training datasets in the configuration file `config_all_dataset.json`.
Then, run this command to train model:

    python train.py --config configs/config_all_dataset.json
    
### Testing

**DTU**

First, download the DTU evaluation dataset from [Yao Yao](https://github.com/YoYo000/MVSNet).
To generate point clouds, users need to install [fusibile](https://github.com/kysucix/fusibile).
Run these commands to build fusibile:

    mkdir build && cd build
    cmake ..
    make
    cp fusibile ../

Then, change the parameters in file `dtu_eval.sh` if necessary and run it to generate reconstruction:

    bash scripts/dtu_eval.sh <path to DTU test set> <pretrained model> <output folder>

To evaluate these reconstructed point clouds, use the evaluation code from the [DTU benchmark website](https://roboimagedata.compute.dtu.dk/?page_id=36). 
We already provide the evaluation code in the evaluation folder. 
The results should be similar to this

|                       | Acc.   | Comp.  | Overall. |
|-----------------------|--------|--------|----------|
| CDS-MVSNet(DTU only, depths=48,32,8, intervals=4.0,1.5,0.75)  | 0.352  | 0.280  | 0.316    |
| CDS-MVSNet(DTU+BlendedMVS, depths=48,32,8, intervals=4.0,1.5,0.75)  | 0.351  | 0.278  | 0.315    |


**Tanks & Temples**

Download the intermediate dataset preprocessed by [Yao Yao](https://github.com/YoYo000/MVSNet).
Note that users should use the short depth range of cameras
Run the evaluation script to produce the point clouds

    bash scripts/tt_eval.sh <path to intermediate set of Tanks&Temples> <pretrained model>

Submit the results to the [Tanks & Temples benchmark website](https://www.tanksandtemples.org/) to receive the F-score. 
Due to large point clouds generated, user may need a NVIDIA card with high memory.

We made the results publicly available on the leaderboard of Tanks & Temples.

### Citations
If this code is helpful for your work, plese cite this

    @article{giang2021curvature,
      title={Curvature-guided dynamic scale networks for Multi-view Stereo},
      author={Giang, Khang Truong and Song, Soohwan and Jo, Sungho},
      journal={arXiv preprint arXiv:2112.05999},
      year={2021}
    }
    
or

    @inproceedings{
      giang2022curvatureguided,
      title={{CURVATURE}-{GUIDED} {DYNAMIC} {SCALE} {NETWORKS} {FOR} {MULTI}-{VIEW}  {STEREO}},
      author={Khang Truong Giang and Soohwan Song and Sungho Jo},
      booktitle={International Conference on Learning Representations},
      year={2022},
      url={https://openreview.net/forum?id=_Wzj0J2xs2D}
    }
