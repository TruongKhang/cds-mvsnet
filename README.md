# CDS-MVSNet
Our experiments are implemented on a Unix system
### Requirements
    conda create -n venv python=3.6
    conda activate venv
    conda install pytorch==1.6.0 torchvision cudatoolkit=10.1 -c pytorch
    # using pip to install required packages if there are any errors

### Training
Our model is trained on DTU dataset. All training parameters are configured in a configuration file `*.json`.
The training datasets including DTU and BlendedMVS are provided by [Yao Yao](https://github.com/YoYo000/MVSNet). 
Change the directory of training dataset in the configuration file.
Then, run this command to train model:

    python train.py --config config_all_dataset.json
    
### Testing
##DTU
First, download the DTU evaluation dataset from [Yao Yao](https://github.com/YoYo000/MVSNet).
To generate point clouds, users need to install [fusibile](https://github.com/kysucix/fusibile). We already provide its source code in our folder.
Run these commands to build fusibile:

    mkdir build && cd build
    cmake ..
    make
    cp fusibile ../

Then, change the parameters in file `dtu_eval.sh` and run it to generate reconstruction:

    bash dtu_eval.sh

To evaluate these reconstructed point clouds, use the evaluation code from the [DTU benchmark website](https://roboimagedata.compute.dtu.dk/?page_id=36).
The result should be similar to this

|                       | Acc.   | Comp.  | Overall. |
|-----------------------|--------|--------|----------|
| CDS-MVSNet(depths=48,32,8, intervals=4.0,1.5,0.75)  | 0.351  | 0.278  | 0.315    |


## Tanks & Temples
Download the intermediate dataset preprocessed by [Yao Yao](https://github.com/YoYo000/MVSNet).
Note that users should use the short depth range of cameras
Run the evaluation script to produce the point clouds

    bash tt_eval.sh

Submit the results to the [Tanks & Temples benchmark website](https://www.tanksandtemples.org/) to receive the F-score. 
Due to large point clouds generated, user may need a NVIDIA card with high memory.

We made the results publicly available. The results should be similar to this

| Mean   | Family | Francis | Horse  | Lighthouse | M60    | Panther | Playground | Train |
|--------|--------|---------|--------|------------|--------|---------|------------|-------|
| 61.58  | 78.85  | 63.17   | 53.04  | 61.34	  | 62.63  | 59.06   | 62.28	  | 52.30 |