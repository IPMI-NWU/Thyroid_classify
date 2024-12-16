import os
import numpy as np
npy_dir='/media/wu24/wu24/wu24/Thyroid/unsupervised/data/cnn_feature/densenet169/train'
npy_names=os.listdir(npy_dir)
for npy_name in npy_names:
    npy_path=os.path.join(npy_dir,npy_name)
    data=np.load(npy_path)
    print(data.shape)
