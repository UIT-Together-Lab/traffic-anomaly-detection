python inference.py  --dataset  shanghaitech    \
          --prednet  cyclegan_convlstm    \
          --num_his  4                     \
          --label_level  normal            \
          --gpu      1                     \
          --interpolation  --snapshot_dir  ./outputs_foggy/highway/checkpoints/normal/shanghaitech/prednet_cyclegan_convlstm/model.ckpt-67000 \
          --psnr_dir './outputs_foggy/highway/psnrs/normal/shanghaitech/prednet_cyclegan_convlstm'