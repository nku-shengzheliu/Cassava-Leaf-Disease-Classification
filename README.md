# Cassava-Leaf-Disease-Classification-NKU_428
>This repository is used to record our code in this [competition](https://www.kaggle.com/c/cassava-leaf-disease-classification/overview)


### Dataset analysis
* 图像数量(Number of images)：21397
* 分辨率(image size)：800*600
* 类别分布(class distribution):
![dataset](https://github.com/nku-shengzheliu/Cassava-Leaf-Disease-Classification/blob/master/dataset.PNG)


### Results
| Loss | Trick | Code | Acc in val | Acc in test|
| :------: | :------: | :------: | :------: | :------: |
| Cross Entropy loss | warmup | - | 0.867 | - |
| Cross Entropy loss | warmup, imgaug | - | - | 0.872 |
| Focal loss | warmup, imgaug | - | - | 0.882 |
| Cross Entropy loss |label smooth, warmup, imgaug |[code](https://github.com/nku-shengzheliu/Cassava-Leaf-Disease-Classification/blob/master/train.py) | - | 0.891 |
| [Bi tempered logistic loss](https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss-for-training.html)  |label smooth, warmup, imgaug| [code](https://github.com/nku-shengzheliu/Cassava-Leaf-Disease-Classification/blob/master/train_bi_temp_loss.py) | 0.8851 |0.890|
| [Bi tempered logistic loss](https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss-for-training.html)  |label smooth, warmup, imgaug, freeze bn| [code](https://github.com/nku-shengzheliu/Cassava-Leaf-Disease-Classification/blob/master/train_bi_temp_loss.py) | 0.8917 |0.890|
| - | - | - | - | - |

### 2021.1.13 Methods for future improvement 可以改进的方向 
- [x] Bi-tempered-logistic-loss (Description found [here](https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss-for-training.html)).
- [x] Frozen BN.
- [ ] Training with early stopping criterion.
- [x] N-Fold CV

### 2021.1.14 Methods for future improvement 可以改进的方向 
- [ ] 将backbone中的maxpool改为[softpool](https://github.com/alexandrosstergiou/SoftPool)
- [ ] use [SAM Optimizer](https://github.com/davda54/sam)

### 2021.1.15 Methods for future improvement 可以改进的方向 
- [ ] use label smooth and cutmix may get score of [0.903](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/209065)
- [x] use 5-fold CV

### 2021.1.16 Methods for future improvement 可以改进的方向 
- [ ] CVPR 2019 Cassava Disease Classification挑战赛: [here](https://blog.csdn.net/fendouaini/article/details/93690986)
