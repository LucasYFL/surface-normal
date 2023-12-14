# EECS 442 Final Project
## Surface Normal Estimation
### Yifu Lu, Xinjin Li, Ruiying Yang

Our work mainly based on the paper: Estimating and Exploiting the Aleatoric Uncertainty in Surface Normal Estimation. 
Based on this, we changed the training pipeline to fit efficientNet V2. We rewrote parts of validation that is not efficient and would potentially cause out of memory error, and added the functionality of resuming training if failed. In addition, we chose NYU v2 large dataset as our training data, for which we wrote a dataloader for .mat input files, and added support for ScanNet.

![442Final_Project](https://github.com/LucasYFL/surface-normal/assets/113412059/5e0e1fe5-8e01-4c46-830e-4c788f924286)
![442Final_Project (1)](https://github.com/LucasYFL/surface-normal/assets/113412059/43ad562b-a6e7-4ed5-8852-eedd094a8736)
![442Final_Project (2)](https://github.com/LucasYFL/surface-normal/assets/113412059/6f70d2e1-bd5a-43ec-837b-82ef72266db8)
![442Final_Project (3)](https://github.com/LucasYFL/surface-normal/assets/113412059/185eea3f-0706-4f38-a228-0ea3ed0798eb)
![442Final_Project (4)](https://github.com/LucasYFL/surface-normal/assets/113412059/472273c1-5981-4964-bb71-5d2b51a3166d)
![442Final_Project (5)](https://github.com/LucasYFL/surface-normal/assets/113412059/5d7e3869-f4dd-4bd8-901e-155a9827dc7b)
![442Final_Project (6)](https://github.com/LucasYFL/surface-normal/assets/113412059/600af8f1-ceea-459b-bde4-efc883d1b217)
![442Final_Project (7)](https://github.com/LucasYFL/surface-normal/assets/113412059/eea9f622-9d3c-4812-b3bc-63f65137f877)
## Training

### Step 1. Download dataset

* **NYUv2 (official)**: The official train/test split contains 795/654 images. The dataset can be downloaded from [this link](https://drive.google.com/drive/folders/1Ku25Am69h_HrbtcCptXn4aetjo7sB33F?usp=sharing). Unzip the file `nyu_dataset.zip` under `./datasets`, so that `./datasets/nyu/train` and `./datasets/nyu/test/` exist.

* **NYUv2 (big)**: Please visit [GeoNet](https://github.com/xjqi/GeoNet) to download a larger training set consisting of 30907 images. This is the training set used to train our model.

* **ScanNet:** Please visit [FrameNet](https://github.com/hjwdzh/framenet/tree/master/src) to download ScanNet with ground truth surface normals.

### Step 2. Train

* Check the args in train.py and simply run 
```python
python train.py
```
