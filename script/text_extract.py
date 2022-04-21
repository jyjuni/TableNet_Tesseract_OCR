import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from PIL import Image
except ImportError:
    import Image

import pytesseract as pt
from itertools import repeat
import csv


def text_extract(test_file_path, output_csv_path, tesseract_path, output_text_path):
	pt.pytesseract.tesseract_cmd = tesseract_path
	image = cv2.imread(test_file_path)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

	# Repair horizontal table lines 
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

	# Remove horizontal lines
	horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (55,2))
	detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
	cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	for c in cnts:
	    cv2.drawContours(image, [c], -1, (255,255,255), 9)

	# Remove vertical lines
	vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,55))
	detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
	cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	for c in cnts:
	    cv2.drawContours(image, [c], -1, (255,255,255), 9)

	data = pt.image_to_string(image, lang='eng',config='--psm 6')
	print(data)
	with open(output_text_path, "w+") as f:
		f.write(data)

	with open(output_text_path, 'r') as in_file:
	    stripped = (line.strip() for line in in_file)
	    lines = (line.split("|") for line in stripped if line)
	    with open(output_csv_path, 'w') as out_file:
	        writer = csv.writer(out_file)
	        writer.writerow((' ', ' '))
	        writer.writerows(lines)


	