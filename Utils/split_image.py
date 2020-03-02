import cv2
import os
import math
from xml_process import get_size_info, create_xml_file
from box_process import calculate_area

# standard input https://github.com/pjreddie/darknet/issues/901
standard_input_width = 832
standard_input_height = 832

start_case = [
	[0, 0],
	[416, 0],
	[0, 416],
	[416, 416]]
	
ori_image_dir = "../../Datasets/TestImages"
image_dir = "../../Datasets/TestSplitImages"


def create_new_img_anno_pair(path):
    ori_img = cv2.imread(path)
    ori_img_width, ori_img_height = ori_img.shape[1], ori_img.shape[0]
    for i in range(len(start_case)):
        case = start_case[i]
        ori_x_start, ori_y_start = case[0], case[1]
        num_x_tiles, num_y_tiles = math.ceil(ori_img_width/standard_input_width), math.ceil(ori_img_height/standard_input_height)
        for x_index in range(num_x_tiles):
            x_start = ori_x_start + x_index * standard_input_width
            x_end = x_start + standard_input_width
            residual_x = ori_img_width - x_end
            # print('x {}'.format(x_index), residual_x)
            if residual_x < 0:
                x_end = ori_img_width
                # print('before x ', residual_x, x_end, ori_img_width)
            # print('after x', residual_x, x_end, ori_img_width)
            for y_index in range(num_y_tiles):
                y_start = ori_y_start + y_index * standard_input_height
                y_end = y_start + standard_input_height
                residual_y = ori_img_height - y_end
                # print('y {}'.format(y_index), residual_y)
                if residual_y < 0:
                    y_end = ori_img_height
                    # print('before y', residual_y, y_end, ori_img_height)
                # print('after y', residual_y, y_end, ori_img_height)
                # boxB = [x_start, y_start, x_end, y_end]
                print('x_start', x_start, 'x_end', x_end, 'y_start', y_start, 'y_end', y_end)
                new_img = ori_img[y_start:y_end, x_start: x_end].copy()
                # cv2.imread(new_img)
                img_name = (path.split('/'))[-1]
                file_name = img_name[:-4] + '{}_{}'.format(x_index, y_index) + '.jpg'
                path_name = os.path.join(image_dir, file_name)
                cv2.imwrite(path_name, new_img)
if __name__ == "__main__":
    list_img = os.listdir(ori_image_dir)
    print("Number img: ", len(list_img), list_img)
    for image in list_img:
        img_path = os.path.join(ori_image_dir, image)
        # split_image = cv2.imread(img_path)
        if os.path.isfile(img_path):
			# print(img_path)
            create_new_img_anno_pair(img_path)