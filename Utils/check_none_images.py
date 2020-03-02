import os
import cv2

img_path = './Datasets/AugTrainImages'
anno_path = './Datasets/AugTrainAnnotations'

def CheckImages():
    none_images = []
    images = os.listdir(img_path)
    for image in images:
        image_value = cv2.imread(os.path.join(img_path, image))
        # print(image)
        if image_value is None:
            none_images.append(image)
    return none_images

def RemoveNoneImagesAndAnnotations(list_none_images):
    images = os.listdir(img_path)
    annotations = os.listdir(anno_path)
    list_none_anno = []
    for i in range(len(list_none_images)):
        none_anno = list_none_images[i][:-4] + '.xml'
        list_none_anno.append(none_anno)
    # images list and annotations list have same amount of files
    print(len(list_none_anno))
    print(len(list_none_images))
    print(len(images))
    print(len(annotations))
    for i in range(len(list_none_images)):
        for j in range(len(images)):
            if list_none_images[i] == images[j] and list_none_images[i] in images:
                os.remove("{}".format(os.path.join(img_path,images[j])))
                # print('here')
            if list_none_anno[i] == annotations[j] and list_none_anno[i] in annotations:
                os.remove("{}".format(os.path.join(anno_path, annotations[j])))


if __name__ == '__main__':
    none_images = CheckImages()
    # print(none_images)
    RemoveNoneImagesAndAnnotations(none_images)

