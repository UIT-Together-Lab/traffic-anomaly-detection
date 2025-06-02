python inference.py  --dataset  shanghaitech    \
          --prednet  cyclegan_convlstm    \
          --num_his  4                     \
          --label_level  normal            \
          --gpu      3                     \
          --interpolation  --snapshot_dir  ./outputs_bike_roundabout/checkpoints/normal/shanghaitech/prednet_cyclegan_convlstm/model.ckpt-64000 \
          --psnr_dir './outputs_bike_roundabout/psnrs/normal/shanghaitech/prednet_cyclegan_convlstm'