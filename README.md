# Cassava-Leaf-Disease-Classification-NKU_428
>This repository is used to record our code in this [competition](https://www.kaggle.com/c/cassava-leaf-disease-classification/overview)

> Our team name: NKU_428

### Dataset analysis
* 图像数量(Number of images)：21397
* 分辨率(image size)：800*600
* 类别分布(class distribution):
![dataset](https://github.com/nku-shengzheliu/Cassava-Leaf-Disease-Classification/blob/master/dataset.PNG)

### 2021.1.13 Methods for future improvement 可以改进的方向 
1. Bi-tempered-logistic-loss (Description found [here](https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss-for-training.html)).
2. Frozen BN.
3. Training with early stopping criterion.
4. N-Fold CV

### 2021.1.14 Methods for future improvement 可以改进的方向 
1. 将backbone中的maxpool改为[softpool](https://github.com/alexandrosstergiou/SoftPool)
2. use [SAM Optimizer](https://github.com/davda54/sam)

### Results
| Loss | Trick | Code | Acc in val | Acc in test|
| :------: | :------: | :------: | :------: | :------: |
| Cross Entropy loss | warmup | - | 0.867 | - |
| Cross Entropy loss | warmup, imgaug | - | - | 0.872 |
| Focal loss | warmup, imgaug | - | - | 0.882 |
| Cross Entropy loss |label smooth, warmup, imgaug |[code](https://github.com/nku-shengzheliu/Cassava-Leaf-Disease-Classification/blob/master/train.py) | - | 0.891 |
| [Bi tempered logistic loss](https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss-for-training.html)  |label smooth, warmup, imgaug| [code](https://github.com/nku-shengzheliu/Cassava-Leaf-Disease-Classification/blob/master/train_bi_temp_loss.py) | - |-|
| - | - | - | - | - |