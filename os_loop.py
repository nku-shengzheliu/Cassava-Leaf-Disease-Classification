import os
for i in range(5):
    os.system("python train_btloss_cv.py -fold_index" + " " + str(i))
