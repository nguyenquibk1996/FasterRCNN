import os

original_image_path = '../../Datasets/OriginalImages'
original_anno_path = '../../Datasets/Annotations'

original_list = os.listdir(original_image_path)


for author in original_list:
    image_list = []
    anno_list = []
    image_author_path = os.path.join(original_image_path, author)
    anno_author_path = os.path.join(original_anno_path, author)
    for image in os.listdir(image_author_path):
        image_non_format = image[:-4]
        image_list.append(image_non_format)
    for anno in os.listdir(anno_author_path):
        anno_non_format = anno[:-4]
        anno_list.append(anno_non_format)
    different_value_image = list(set(anno_list)-set(image_list))
    print('author:', author)
    print('different_value', different_value_image)