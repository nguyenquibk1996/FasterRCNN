def calculate_area(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxA_area = (boxA[3] - boxA[1]) * (boxA[2] - boxA[0])
	
	overlap_area = int(interArea * 1000/ boxA_area)
	return overlap_area