#!/usr/bin/env bash
TESTPATH="/mnt/sdb1/khang/dtu_dataset/test"
TESTLIST="lists/dtu/test.txt"
CKPT_FILE="saved/models/CDS-MVSNet/1112_095359/checkpoint-epoch19.pth"
python test.py --dataset dtu --batch_size 1 --testpath $TESTPATH --testlist $TESTLIST --resume $CKPT_FILE --outdir gauss_curv_outputs --interval_scale 1.06 --num_view 5 --numdepth 192 --max_h 1152 --max_w 1536 --filter_method gipuma --disp_threshold 0.1 --num_consistent 2
