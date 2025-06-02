python inference.py  --dataset  shanghaitech    \
          --prednet  cyclegan_convlstm    \
          --num_his  4                     \
          --label_level  normal            \
          --gpu      2                     \
          --interpolation  --snapshot_dir  ./outputs_foggy/railway_inspection/checkpoints/normal/shanghaitech/prednet_cyclegan_convlstm/model.ckpt-75000 \
          --psnr_dir './outputs_foggy/railway_inspection/psnrs/normal/shanghaitech/prednet_cyclegan_convlstm'