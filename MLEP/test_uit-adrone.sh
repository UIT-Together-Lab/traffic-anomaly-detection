python inference.py  --dataset  shanghaitech    \
          --prednet  cyclegan_convlstm    \
          --num_his  4                     \
          --label_level  normal            \
          --gpu      3                     \
          --interpolation  --snapshot_dir  ./outputs_uit-adrone/checkpoints/normal/shanghaitech/prednet_cyclegan_convlstm/model.ckpt-75000 \
          --psnr_dir './outputs_uit-adrone/psnrs/normal/shanghaitech/prednet_cyclegan_convlstm'