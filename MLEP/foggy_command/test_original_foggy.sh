python inference.py  --dataset  shanghaitech    \
          --prednet  cyclegan_convlstm    \
          --num_his  4                     \
          --label_level  normal            \
          --gpu      0                     \
          --interpolation                  \
          --snapshot_dir  ./outputs_original/bike_roundabout/checkpoints/normal/shanghaitech/prednet_cyclegan_convlstm/model.ckpt-75000 \
          --psnr_dir './outputs_original/bike_roundabout/psnrs/normal/shanghaitech/prednet_cyclegan_convlstm'
          
          
python inference.py  --dataset  shanghaitech    \
          --prednet  cyclegan_convlstm    \
          --num_his  4                     \
          --label_level  normal            \
          --gpu      1                     \
          --interpolation  --snapshot_dir  ./outputs_original/vehicle_roundabout/checkpoints/normal/shanghaitech/prednet_cyclegan_convlstm/model.ckpt-75000 \
          --psnr_dir './outputs_original/vehicle_roundabout/psnrs/normal/shanghaitech/prednet_cyclegan_convlstm'
          
python inference.py  --dataset  shanghaitech    \
          --prednet  cyclegan_convlstm    \
          --num_his  4                     \
          --label_level  normal            \
          --gpu      2                     \
          --interpolation  --snapshot_dir  ./outputs_original/recheck/railway_inspection/checkpoints/normal/shanghaitech/prednet_cyclegan_convlstm/model.ckpt-75000 \
          --psnr_dir './outputs_original/recheck/railway_inspection/psnrs/normal/shanghaitech/prednet_cyclegan_convlstm'
          
python inference.py  --dataset  shanghaitech    \
          --prednet  cyclegan_convlstm    \
          --num_his  3                     \
          --label_level  normal            \
          --gpu      0                     \
          --interpolation  --snapshot_dir  ./outputs_original/highway/checkpoints/normal/shanghaitech/prednet_cyclegan_convlstm/model.ckpt-75000 \
          --psnr_dir './outputs_original/highway/psnrs/normal/shanghaitech/prednet_cyclegan_convlstm'
          
python inference.py  --dataset  shanghaitech    \
          --prednet  cyclegan_convlstm    \
          --num_his  4                     \
          --label_level  normal            \
          --gpu      0                     \
          --interpolation  --snapshot_dir  ./outputs_original/crossroads/checkpoints/normal/shanghaitech/prednet_cyclegan_convlstm/model.ckpt-75000 \
          --psnr_dir './outputs_original/crossroads/psnrs/normal/shanghaitech/prednet_cyclegan_convlstm'

python inference.py  --dataset  shanghaitech    \
          --prednet  cyclegan_convlstm    \
          --num_his  4                     \
          --label_level  normal            \
          --gpu      0                     \
          --interpolation  --snapshot_dir  ./outputs_uit-adrone_original_foggy/checkpoints/normal/shanghaitech/prednet_cyclegan_convlstm/model.ckpt-75000 \
          --psnr_dir './outputs_uit-adrone_original_foggy/psnrs/normal/shanghaitech/prednet_cyclegan_convlstm'
