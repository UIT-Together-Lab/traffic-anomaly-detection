python train_normal_annotation.py --dataset  shanghaitech    \
         --prednet  cyclegan_convlstm    \
         --batch    2                    \
         --num_his  4                    \
         --label_level  normal           \
         --gpu      0                    \
         --iters    80000  --output_dir  ./outputs_foggy/bike_roundabout