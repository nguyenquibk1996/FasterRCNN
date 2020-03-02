import xml.etree.ElementTree as ET
import os
import shutil

# annotation_dir = "./Datasets/AugAnnotations"
# img_dir = "./Datasets/AugImages"
img_dir = "./Datasets/AugTrainImages"
annotation_dir = "./Datasets/AugTrainAnnotations"

list_wrong_anno = []

def check_bounding_box(share_name):
	print(share_name)
	anno_path = os.path.join(annotation_dir, share_name)
	ori_tree = ET.parse(anno_path)
	ori_root = ori_tree.getroot()
	x_min, y_min, x_max, y_max = 0, 0, 0, 0
	width, height = 0, 0
	for child in ori_root.iter():
		if child.tag == "object":
			for sub_att in child.iter():
				if sub_att.tag == "bndbox":
					for coor in sub_att:
						if coor.tag == "xmin":
							x_min = int(coor.text)
						if coor.tag == "xmax":
							x_max = int(coor.text)
						if coor.tag == "ymin":
							y_min = int(coor.text)
						if coor.tag == "ymax":
							y_max = int(coor.text)
		if child.tag == "size":
			for sub_att in child:
				if sub_att.tag == "width":
					width = int(sub_att.text)
				if sub_att.tag == "height":
					height = int(sub_att.text)
	if y_max > height:
		print("height: ", height, " y_max: ", y_max)
		print("This anno ", share_name, " is wrong")
		list_wrong_anno.append(share_name)
	if x_max > width:
		print("width: ", width, " x_max: ", x_max)
		print("This anno ", share_name, " is wrong")
		list_wrong_anno.append(share_name)

if __name__ == "__main__":
	list_anno = next(os.walk(annotation_dir))[2]
	list_anno = sorted(list_anno)
	for anno in list_anno:
		check_bounding_box(anno)
	print("number wrong anno: ", len(list_wrong_anno))
	# for wrong_anno in list_wrong_anno:
	# 	old_anno_path = os.path.join(annotation_dir, wrong_anno)
	# 	old_img_path = os.path.join(img_dir, wrong_anno[:-4] + ".jpg")
	# 	# new_anno_path = os.path.join(".", wrong_anno)
	# 	# new_img_path = os.path.join(".", wrong_anno[:-4] + ".jpg")
	# 	# shutil.move(old_anno_path, new_anno_path)
	# 	# shutil.move(old_img_path, new_img_path)
	# 	os.remove(old_anno_path)
	# 	os.remove(old_img_path)