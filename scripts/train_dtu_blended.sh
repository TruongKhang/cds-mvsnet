#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

main_cfg_path=$1 #"configs/megadepth_train.py"

n_gpus=4
torch_num_workers=8

# python -u ./pl_train.py --config "configs/config_dtu.json" \
#     --n_gpus=${n_gpus} \
#     --batch_size=1 --val_batch_size=1 --num_workers=${torch_num_workers} \
#     --max_epochs=15 # --ckpt_path="logs/tb_logs/dtu/version_2/checkpoints/last.ckpt"

python -u ./pl_train.py --config "configs/config_blended.json" \
    --n_gpus=${n_gpus} \
    --batch_size=1 --val_batch_size=1 --num_workers=${torch_num_workers} \
    --max_epochs=25 --ckpt_path="logs/tb_logs/dtu/version_67/checkpoints/last.ckpt" #"logs/tb_logs/dtu/version_55/checkpoints/last.ckpt" #"logs/tb_logs/dtu/version_23/checkpoints/last.ckpt"

# python -u ./pl_train.py --config ${main_cfg_path} \
#     --n_gpus=${n_gpus} \
#     --batch_size=1 --val_batch_size=1 --num_workers=${torch_num_workers} \
#     --max_epochs=10 --ckpt_path="logs/tb_logs/finetuning_blendedmvs/version_37/checkpoints/last.ckpt"


# python -u ./pl_train.py --config ${main_cfg_path} \
#     --n_gpus=${n_gpus} \
#     --batch_size=1 --val_batch_size=1 --num_workers=${torch_num_workers} \
#     --max_epochs=30 --ckpt_path="pretrained/both_dtu_blended/cds_mvsnet.ckpt"
