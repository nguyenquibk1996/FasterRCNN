import os
import math

train_img_path = '../../Datasets/TrainImages'
test_img_path = '../../Datasets/TestImages'
train_anno_path = '../../Datasets/TrainAnnotations'
test_anno_path = '../../Datasets/TestAnnotations'

main_path = '../../Datasets/Main'

if not os.path.isdir(main_path):
    os.mkdir(main_path)

trainval_img = os.listdir(train_img_path)

value = math.floor(len(trainval_img)*0.8)
train_img = trainval_img[:value]
val_img = trainval_img[value:]

for i in range(len(train_img)):
    train_img[i] = train_img[i][:-4]
    stamp_train_file = open('../../Datasets/Main/Stamp_train.txt', 'a+')
    stamp_train_file.writelines(train_img[i])
    stamp_train_file.write(' -1'+'\n')
    train_file = open('../../Datasets/Main/train.txt', 'a+')
    train_file.writelines(train_img[i])
    train_file.write('\n')
stamp_train_file.close()

for i in range(len(val_img)):
    val_img[i] = val_img[i][:-4]
    stamp_val_file = open('../../Datasets/Main/Stamp_val.txt', 'a+')
    stamp_val_file.writelines(val_img[i])
    stamp_val_file.write(' -1'+'\n')
    val_file = open('../../Datasets/Main/val.txt', 'a+')
    val_file.writelines(val_img[i])
    val_file.write('\n')
stamp_val_file.close()

#Generate trainval text file
train_img = os.listdir(train_img_path)
for i in range(len(train_img)):
    train_img[i] = train_img[i][:-4]
    stamp_trainval_file = open('../../Datasets/Main/Stamp_trainval.txt', 'a+')
    stamp_trainval_file.writelines(train_img[i])
    stamp_trainval_file.write(' -1'+'\n')
    trainval_file = open('../../Datasets/Main/trainval.txt', 'a+')
    trainval_file.writelines(train_img[i])
    trainval_file.write('\n')
stamp_trainval_file.close()

#Generate test file
test_img = os.listdir(test_img_path)
# print(test_img)
for i in range(len(test_img)):
    test_img[i] = test_img[i][:-4]
    stamp_test_file = open('../../Datasets/Main/Stamp_test.txt', 'a+')
    stamp_test_file.writelines(test_img[i])
    stamp_test_file.write(' -1'+'\n')
    test_file = open('../../Datasets/Main/test.txt', 'a+')
    test_file.writelines(test_img[i])
    test_file.write('\n')
# test_file.close()