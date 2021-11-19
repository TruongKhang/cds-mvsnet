#!/usr/bin/env bash
TESTPATH="/mnt/sdb/khang/dtu_dataset/test"
TESTLIST="lists/dtu/test.txt"
CKPT_FILE=$1
OUTDIR=$2
python test.py --dataset dtu --batch_size 1 --testpath $TESTPATH --testlist $TESTLIST --resume $CKPT_FILE --outdir $OUTDIR --interval_scale 1.06 --num_view 5 --numdepth 192 --max_h 1152 --max_w 1536 --filter_method gipuma --disp_threshold 0.1 --num_consistent 2
