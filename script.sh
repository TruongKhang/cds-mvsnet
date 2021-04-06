#scp -P 10022 khang@143.248.135.115:~/project/cascade-stereo/CasMVSNet/outputs/*.ply outputs/
#matlab -nodisplay -nosplash -nodesktop -r "run('/home/khangtg/Documents/lab/code/cascade-stereo/CasMVSNet/evaluations/dtu/BaseEvalMain_web.m');exit;" | tail -n +11
#matlab -nodisplay -nosplash -nodesktop -r "run('/home/khangtg/Documents/lab/code/cascade-stereo/CasMVSNet/evaluations/dtu/ComputeStat_web.m');exit;" | tail -n +11
#mv outputs/eval_out/TotalStat_mvsnet_Eval_.txt results/eval_gipuma.txt
#rm -rf outputs/eval_out
#rm outputs/*.ply
#rm -rf outputs/scan*/points_mvsnet/con*
#####################################
./test.sh  checkpoints/final_model.ckpt --outdir outputs  --interval_scale 1.06  --filter_method gipuma --numdepth 192 --ndepths 48,24,8 --depth_inter_r 4,3,1 --prob_threshold 0.8 --disp_threshold 0.25 --num_consistent 4
python move.py outputs
matlab -nodisplay -nosplash -nodesktop -r "run('/home/khangtg/Documents/lab/seq-prob-mvs/evaluations/dtu/BaseEvalMain_web.m');exit;" | tail -n +11
matlab -nodisplay -nosplash -nodesktop -r "run('/home/khangtg/Documents/lab/seq-prob-mvs/evaluations/dtu/ComputeStat_web.m');exit;" | tail -n +11
mv outputs/eval_out/TotalStat_mvsnet_Eval_.txt results/eval_0.txt
rm -rf outputs/eval_out
rm outputs/*.ply
rm -rf outputs/scan*/points_mvsnet/con*
#scp -P 10022 -r khang@143.248.135.115:~/project/seq-prob-mvs/outputs .
./test.sh  checkpoints/final_model.ckpt --outdir outputs  --interval_scale 1.06  --filter_method gipuma --numdepth 192 --ndepths 48,24,8 --depth_inter_r 4,3,1 --prob_threshold 0.8 --disp_threshold 0.3 --num_consistent 4
python move.py outputs
matlab -nodisplay -nosplash -nodesktop -r "run('/home/khangtg/Documents/lab/seq-prob-mvs/evaluations/dtu/BaseEvalMain_web.m');exit;" | tail -n +11
matlab -nodisplay -nosplash -nodesktop -r "run('/home/khangtg/Documents/lab/seq-prob-mvs/evaluations/dtu/ComputeStat_web.m');exit;" | tail -n +11
mv outputs/eval_out/TotalStat_mvsnet_Eval_.txt results/eval_1.txt
mv outputs outputs_old
#rm -rf outputs/eval_out
#rm outputs/*.ply
#rm -rf outputs/scan*/points_mvsnet/con*
######
scp -P 10022 -r khang@143.248.135.115:~/project/seq-prob-mvs/outputs .
./test.sh  checkpoints/final_model.ckpt --outdir outputs  --interval_scale 1.06  --filter_method gipuma --numdepth 192 --ndepths 48,24,8 --depth_inter_r 4,3,1 --prob_threshold 0.8 --disp_threshold 0.3 --num_consistent 4
python move.py outputs
matlab -nodisplay -nosplash -nodesktop -r "run('/home/khangtg/Documents/lab/seq-prob-mvs/evaluations/dtu/BaseEvalMain_web.m');exit;" | tail -n +11
matlab -nodisplay -nosplash -nodesktop -r "run('/home/khangtg/Documents/lab/seq-prob-mvs/evaluations/dtu/ComputeStat_web.m');exit;" | tail -n +11
mv outputs/eval_out/TotalStat_mvsnet_Eval_.txt results/eval_2.txt
#rm -rf outputs/eval_out
#rm outputs/*.ply
#rm -rf outputs/scan*/points_mvsnet/con*
######
