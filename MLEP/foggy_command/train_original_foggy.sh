python train_normal_annotation.py --dataset  shanghaitech    \
         --prednet  cyclegan_convlstm    \
         --batch    2                    \
         --num_his  4                    \
         --label_level  normal           \
         --gpu      0                    \
         --iters    80000  --output_dir  ./outputs_original/bike_roundabout
         
         
python train_normal_annotation.py --dataset  shanghaitech    \
         --prednet  cyclegan_convlstm    \
         --batch    2                    \
         --num_his  4                    \
         --label_level  normal           \
         --gpu      0                    \
         --iters    80000  --output_dir  ./outputs_original/vehicle_roundabout
         
python train_normal_annotation.py --dataset  shanghaitech    \
         --prednet  cyclegan_convlstm    \
         --batch    2                    \
         --num_his  4                    \
         --label_level  normal           \
         --gpu      0                    \
         --iters    80000  --output_dir  ./outputs_original/recheck/railway_inspection
         
python train_normal_annotation.py --dataset  shanghaitech    \
         --prednet  cyclegan_convlstm    \
         --batch    2                    \
         --num_his  4                    \
         --label_level  normal           \
         --gpu      0                    \
         --iters    80000  --output_dir  ./outputs_original/highway
         
python train_normal_annotation.py --dataset  shanghaitech    \
         --prednet  cyclegan_convlstm    \
         --batch    2                    \
         --num_his  4                    \
         --label_level  normal           \
         --gpu      0                    \
         --iters    80000  --output_dir  ./outputs_original/crossroads