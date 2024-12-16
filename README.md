#  Histopathology Image Diagnosis via Whole Tissue Slide Level Supervision

# Envs
- Linux
- Python>=3.7
- CPU or NVIDIA GPU + CUDA CuDNN
- kfbslide

# datasets:  Private dataset


# Train

```
python ./train/train_densenet169_pce.py --weight 10.0 --save_dir ./path/to/save/dir


