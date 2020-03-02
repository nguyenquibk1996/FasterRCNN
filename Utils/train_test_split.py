import os
import shutil
import math

anno_path = './Datasets/Annotations'
img_path = './Datasets/Images'
train_img_path = './Datasets/TrainImages'
test_img_path = './Datasets/TestImages'
train_anno_path = './Datasets/TrainAnnotations'
test_anno_path = './Datasets/TestAnnotations'

#Check and make directories
if not os.path.isdir(train_img_path):
    os.mkdir(train_img_path)
if not os.path.isdir(train_anno_path):
    os.mkdir(train_anno_path)
if not os.path.isdir(test_img_path):
    os.mkdir(test_img_path)
if not os.path.isdir(test_anno_path):
    os.mkdir(test_anno_path)

#Get files and split train, test
anno_files = os.listdir(anno_path)
img_files = os.listdir(img_path)
print('Number of Annotations:', len(anno_files))
print('Number of Images:', len(img_files))

for i in range(len(img_files)):
    anno_files[i] = './Datasets/Annotations/' + anno_files[i]
    img_files[i] = './Datasets/Images/' + img_files[i]

value = math.floor(len(img_files)*0.8)
train_images = img_files[:value]
test_images = img_files[value:]
train_annotations = anno_files[:value]
test_annotations = anno_files[value:]
print('-------------------------')
print('Number of Train Images:', len(train_images))
print('Number of Test Images:', len(test_images))
print('Number of Train Annotations:', len(train_annotations))
print('Number of Test Annotations:', len(test_annotations))
# print(len(os.listdir(train_img_path)))
# print(type(0))
#Copy files to train, test folders
if len(os.listdir(train_img_path)) > 0 or len(os.listdir(train_anno_path)) > 0:
    print('Files existed!')
else:
    for i in range(len(train_images)):
        shutil.copy2(train_images[i], train_img_path)
        shutil.copy2(train_annotations[i], train_anno_path)

if len(os.listdir(test_img_path)) > 0 or len(os.listdir(test_anno_path)) > 0:
    print('Files existed!')
else: 
    for i in range(len(test_images)):
        shutil.copy2(test_images[i], test_img_path)
        shutil.copy2(test_annotations[i], test_anno_path)
    