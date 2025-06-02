python inference.py  --dataset  shanghaitech    \
          --prednet  cyclegan_convlstm    \
          --num_his  4                     \
          --label_level  normal            \
          --gpu      3                     \
          --interpolation  --snapshot_dir  ./outputs_vehicle_roundabout/checkpoints/normal/shanghaitech/prednet_cyclegan_convlstm/model.ckpt-35000 \
          --psnr_dir './outputs_vehicle_roundabout/psnrs/normal/shanghaitech/prednet_cyclegan_convlstm'