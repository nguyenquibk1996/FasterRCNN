import xml.etree.ElementTree as ET
from collections import namedtuple
import cv2
import os
import shutil
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
ori_annotation_dir = "../../Datasets/TestAnnotations"
image_dir = "../../Datasets/AugTestImages"
annotation_dir = "../../Datasets/AugTestAnnotations"

log_info = open("log_info.txt","w")

if not os.path.isdir(ori_image_dir):
	print("This is not images directory")
	exit()

if not os.path.isdir(ori_annotation_dir):
	print("This is not annotation directory")
	exit()

if not os.path.isdir(image_dir):
	os.mkdir(image_dir)
	print("Created img directory")
	
if not os.path.isdir(annotation_dir):
	os.mkdir(annotation_dir)
	print("Created annotation directory")

def create_new_img_anno_pair(share_name):
	anno_path = os.path.join(ori_annotation_dir, share_name+".xml")
	print(share_name)
	log_info.write("Create from: " +  str(img_path) +  " and " + str(anno_path) + "\n")
	list_coordinate, w_h_d = get_size_info(anno_path)
	ori_img_width, ori_img_height = w_h_d[0], w_h_d[1]
	if len(list_coordinate) == 0:
		log_info.write("Do nothing with this image \n")
		return
	ori_img = cv2.imread(os.path.join(ori_image_dir, share_name + ".jpg"))
	for coordinate in list_coordinate:
		to_be_created = [1, 1, 1, 1]
		boxA = coordinate
		x_min, y_min, x_max, y_max = boxA[0], boxA[1], boxA[2], boxA[3] 
		for i in range(len(start_case)):
			case = start_case[i]
			ori_x_start, ori_y_start = case[0], case[1]
			num_x_tiles, num_y_tiles = math.ceil(ori_img_width/standard_input_width), math.ceil(ori_img_height/standard_input_height)
			for x_index in range(num_x_tiles):
				x_start = ori_x_start + x_index * standard_input_width
				x_end = x_start + standard_input_width
				residual_x = ori_img_width - x_end
				if residual_x < 0:
					x_end = ori_img_width
				for y_index in range(num_y_tiles):
					y_start = ori_y_start + y_index * standard_input_height
					y_end = y_start + standard_input_height
					residual_y = ori_img_height - y_end
					if residual_y < 0:
						y_end = ori_img_height
						boxB = [x_start, y_start, x_end, y_end]
						overlap_area = calculate_area(boxA, boxB)
						if overlap_area != 0 and overlap_area != 1000:
							to_be_created[i] = 0
					boxB = [x_start, y_start, x_end, y_end]
					overlap_area = calculate_area(boxA, boxB)
					if overlap_area != 0 and overlap_area != 1000:
						to_be_created[i] = 0
		log_info.write(str(to_be_created) + "\n")
		for i in range(len(to_be_created)):
			if to_be_created[i] == 1:
				case = start_case[i]
				seal_part = None
				log_info.write("case to be created: " + str(case) + "\n")
				ori_x_start, ori_y_start = case[0], case[1]
				num_x_tiles, num_y_tiles = math.floor(ori_img_width/standard_input_width), math.floor(ori_img_height/standard_input_height)
				new_x_min, new_x_max, new_y_min, new_y_max = 0, 0, 0, 0
				for x_index in range(num_x_tiles):
					x_start = ori_x_start + x_index * standard_input_width
					x_end = x_start + standard_input_width
					residual_x = ori_img_width - x_end
					if residual_x < 0:
						x_end = ori_img_width
					for y_index in range(num_y_tiles):
						y_start = ori_y_start + y_index * standard_input_height
						y_end = y_start + standard_input_height
						residual_y = ori_img_height - y_end
						if residual_y < 0:
							y_end = ori_img_height
							boxB = [x_start, y_start, x_end, y_end]
							overlap_area = calculate_area(boxA, boxB)
							if overlap_area == 1000:
								new_x_min = x_min - x_start
								new_y_min = y_min - y_start							
								new_x_max = x_max - x_start							
								new_y_max = y_max - y_start							
								new_img_name = share_name + "_" + str(x_start) + "_" + str(y_start) +"_" + str(new_x_min) + "_" + str(new_y_min) + "_" + str(new_x_max) + "_" + str(new_y_max) + ".jpg"
								new_img_path = os.path.join(image_dir, new_img_name)
								new_img = ori_img[y_start:y_end, x_start: x_end].copy()
								seal_part = new_img[new_y_min: new_y_max, new_x_min: new_x_max].copy()
								start_point = [x_start, y_start]
								new_bbox = [new_x_min, new_y_min, new_x_max, new_y_max]
								new_w_h = [x_end - x_start, y_end - y_start]
								create_xml_file(share_name, img_name, start_point, new_bbox, new_w_h)
								cv2.imwrite(new_img_path, new_img)
						else:
							boxB = [x_start, y_start, x_end, y_end]
							overlap_area = calculate_area(boxA, boxB)
							if overlap_area == 1000:
								new_x_min = x_min - x_start
								new_y_min = y_min - y_start							
								new_x_max = x_max - x_start							
								new_y_max = y_max - y_start		
								new_img_name = share_name + "_" + str(x_start) + "_" + str(y_start) +"_" + str(new_x_min) + "_" + str(new_y_min) + "_" + str(new_x_max) + "_" + str(new_y_max) + ".jpg"
								new_img_path = os.path.join(image_dir, new_img_name)
								new_img = ori_img[y_start:y_end, x_start: x_end].copy()
								seal_part = new_img[new_y_min: new_y_max, new_x_min: new_x_max].copy()
								start_point = [x_start, y_start]
								new_bbox = [new_x_min, new_y_min, new_x_max, new_y_max]
								new_w_h = [x_end - x_start, y_end - y_start]
								create_xml_file(share_name, img_name, start_point, new_bbox, new_w_h)
								cv2.imwrite(new_img_path, new_img)
				log_info.write("Start to create new image \n")
				
				for x_index in range(num_x_tiles):
					x_start = ori_x_start + x_index * standard_input_width
					x_end = x_start + standard_input_width
					residual_x = ori_img_width - x_end
					if residual_x < 0:
						x_end = ori_img_width
					for y_index in range(num_y_tiles):
						y_start = ori_y_start + y_index * standard_input_height
						y_end = y_start + standard_input_height
						residual_y = ori_img_height - y_end
						if residual_y < 0:
							y_end = ori_img_height
						boxB = [x_start, y_start, x_end, y_end]
						overlap_area = calculate_area(boxA, boxB)
						try: 
							if overlap_area != 1000 and seal_part is not None:
								height_seal, width_seal = seal_part.shape[0], seal_part.shape[1]
								if height_seal < y_end -y_start and width_seal < x_end - x_start: 
									log_info.write("x start: " + str(x_start) + " x end: " + str(x_end) + " y start: " + str(y_start) + " y end: " + str(y_end) + "\n")
									new_img_name = share_name + "_" + str(x_start) + "_" + str(y_start) +"_" + str(new_x_min) + "_" + str(new_y_min) + "_" + str(new_x_max) + "_" + str(new_y_max) + ".jpg"
									new_img_path = os.path.join(image_dir, new_img_name)
									log_info.write(new_img_name)
									log_info.write("\n")
									log_info.write("bouding box: " + str(new_x_min) + "_" + str(new_y_min) + "_" + str(new_x_max) + "_" + str(new_y_max) + "\n")
									new_img = ori_img[y_start:y_end, x_start: x_end].copy()
									new_img[new_y_min: new_y_max, new_x_min: new_x_max] = seal_part
									start_point = [x_start, y_start]
									new_bbox = [new_x_min, new_y_min, new_x_max, new_y_max]
									log_info.write(str(new_bbox) + "\n")
									new_w_h = [x_end - x_start, y_end - y_start]
									create_xml_file(share_name, img_name, start_point, new_bbox, new_w_h)
									cv2.imwrite(new_img_path, new_img)
									log_info.write("Create img: " + str(share_name) + " case: " + str(case)
							 + " x_tile: " + str(x_index) + " y_tile: " + str(y_index) + "\n")
						except Exception as e:
							log_info.write("can not create img: " + str(share_name) + " case: " + str(case)
							 + " x_tile: " + str(x_index) + " y_tile: " + str(y_index) + "\n")
							log_info.write("Exception: " + str(e) + "\n")
												
if __name__ == "__main__":
	list_anno = next(os.walk(ori_annotation_dir))[2]
	list_anno = sorted(list_anno)
	print("Number anno: ", len(list_anno))
	for anno in list_anno:
		share_name = anno[:-4]
		img_name = share_name + ".jpg"
		img_path = os.path.join(ori_image_dir, img_name)
		if os.path.isfile(img_path):
			print(img_path)
			create_new_img_anno_pair(share_name)