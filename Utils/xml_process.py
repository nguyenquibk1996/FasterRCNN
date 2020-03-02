import xml.etree.ElementTree as ET
import os

ori_image_dir = "../../Datasets/TestImages"
ori_annotation_dir = "../../Datasets/TestAnnotations"
image_dir = "../../Datasets/AugTestImages"
annotation_dir = "../../Datasets/AugTestAnnotations"

def get_size_info(anno_path):
	tree = ET.parse(anno_path)
	root = tree.getroot()
	get_attribute = False
	list_coordinate = []
	w_h_d = []
	for child in root:
		if child.tag == "object":
			for sub_att in child:
				if sub_att.tag == "name" and sub_att.text == "Stamp":
					get_attribute = True
				if get_attribute:
					if sub_att.tag == "bndbox":
						bouding_box = []
						for coor in sub_att:
							bouding_box.append(int(coor.text))
						get_attribute = False
						list_coordinate.append(bouding_box)
		if child.tag == "size":
			for sub_att in child:
				w_h_d.append(int(sub_att.text))
	return list_coordinate, w_h_d

def create_xml_file(share_name, img_name, start_point, new_bbox, new_w_h):
	anno_path = os.path.join(ori_annotation_dir, share_name+".xml")
	ori_tree = ET.parse(anno_path)
	ori_root = ori_tree.getroot()
	x_min_new, y_min_new, x_max_new, y_max_new = new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3]
	new_width, new_height = new_w_h[0], new_w_h[1]
	x_start = start_point[0]
	y_start = start_point[1]
	str_to_add = "_" + str(x_start)+ "_" + str(y_start) + "_" + str(x_min_new) + "_" + str(y_min_new)+ "_" + str(x_max_new) + "_" + str(y_max_new)
	for child in ori_root.iter():
		if child.tag == "object":
			for sub_att in child.iter():
				if sub_att.tag == "bndbox":
					print("here")
					for coor in sub_att:
						if coor.tag == "xmin":
							print("old: ", coor.text)
							coor.text =  coor.text.replace(coor.text, str(x_min_new))
							print("new: ", coor.text)
						if coor.tag == "xmax":
							print("old: ", coor.text)
							coor.text =  coor.text.replace(coor.text, str(x_max_new))
							print("new: ", coor.text)
						if coor.tag == "ymin":
							coor.text =  coor.text.replace(coor.text, str(y_min_new))
						if coor.tag == "ymax":
							coor.text =  coor.text.replace(coor.text, str(y_max_new))
		if child.tag == "size":
			for sub_att in child:
				if sub_att.tag == "width":
					sub_att.text = str(new_width)
				if sub_att.tag == "height":
					sub_att.text = str(new_height)
		if child.tag == "filename":
			old_name = child.text
			new_name = old_name + str_to_add
			child.text = new_name 
		if child.tag == "path":
			old_path = child.text[:-4]
			new_path = old_path + str_to_add
			child.text = new_path + ".jpg"
	new_xml_name = share_name + str_to_add + ".xml"
	
	new_xml_path = os.path.join(annotation_dir, new_xml_name)
	print("new xml name: ", new_xml_path)
	ori_tree.write(new_xml_path)