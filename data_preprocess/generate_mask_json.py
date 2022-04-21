import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path


# for debugging purpose only
def show_sample(i):


	df = pd.read_csv("dataset/Fintabnet_data/GT.csv", encoding='utf-8')
	ant = df.iloc[:,3]

	# load JSON
	x = ant[i] #img_0

	# parse x:
	y = json.loads(x)

	n_cols = sum(d["result"]["类型"]=="表头" for d in y["annotations"])
	col_x_mins, col_x_maxs = [float("inf")] * n_cols, [0] * n_cols
	col_y_mins, col_y_maxs = [float("inf")] * n_cols, [0] * n_cols

	return y["annotations"], n_cols   


if __name__ == "__main__":

	directory = './dataset/Fintabnet_data/input/'
	final_col_directory = './dataset/Fintabnet_data/column_mask/'
	final_table_directory = './dataset/Fintabnet_data/table_mask/'
	final_cell_directory = './dataset/Fintabnet_data/cell_mask/'
	max_tables = 0

	df = pd.read_csv("dataset/Fintabnet_data/GT.csv", encoding='utf-8')
	ant = df.iloc[:,3]

	for i in range(len(ant)):
		y = json.loads(ant[i])

		# Find all the non-empty annotation records
		if y:
			filename = f"file{i}"

			# Parse width, height
			width = y['container']["page1"]["width"]
			height = y['container']["page1"]["height"]
			
			# convert pdf to image
			page = convert_from_path(f'{directory}{filename}.pdf', size=(width*5, height*5))[0]
			
			# Create grayscale image array
			col_mask = np.zeros((height, width), dtype=np.int32)
			table_mask = np.zeros((height, width), dtype = np.int32)
			cell_mask = np.zeros((height, width), dtype = np.int32)

			valid = False
			if any("result" not in d for d in y["annotations"]):
				print(f"ERROR: {filename} no result")
				continue
			n_cols = sum(d["result"]["类型"]=="表头" for d in y["annotations"])
			n_tables = sum(d["result"]["类型"]=="表格" for d in y["annotations"])
			print(f"{filename}:{n_cols} columns, {n_tables} tables")
			if not n_cols:
				# n_cols = count_cols(y["annotations"])
				n_cols = 2
				print(f"WARNING: {filename} no header")
			col_x_mins, col_x_maxs = [height] * n_cols, [0] * n_cols
			col_y_mins, col_y_maxs = [width] * n_cols, [0] * n_cols

			# y
			# the result is a Python dictionary:
			for i, d in enumerate(y["annotations"]):
				x_min = int(min(corner[0] for corner in d["points"]))
				x_max = int(max(corner[0] for corner in d["points"]))
				y_min = int(min(corner[1] for corner in d["points"]))
				y_max = int(max(corner[1] for corner in d["points"]))

				if d["result"]["类型"]=="无效数据":
					print(f"{filename} is invalid")
					break
				elif d["result"]["类型"]=="表格": 
					# fill column mask
					for ci in range(n_cols):
						col_mask[col_y_mins[ci]:col_y_maxs[ci], col_x_mins[ci]:col_x_maxs[ci]] = 255

					# fill table mask
					table_mask[y_min:y_max, x_min:x_max] = 255

					# export cropped table image
					table_image = page.crop((d["points"][0][0]*5, d["points"][0][1]*5, d["points"][2][0]*5, d["points"][2][1]*5))
					table_file_name = directory + f"ocr_input/{filename}"
					table_image.save(f"{table_file_name}.png")

					# # export whole table ocr GT
					# generate_table_text(y, i)
					

					# clear col_ arrays
					col_x_mins, col_x_maxs = [height] * n_cols, [0] * n_cols
					col_y_mins, col_y_maxs = [width] * n_cols, [0] * n_cols

				else:
					valid = True
					# fill cell mask
					cell_mask[y_min:y_max, x_min:x_max] = 255
					
					# if len(d["label"].splitlines())>1:
					#     continue
					cell_file_name = f"{final_cell_directory}/{filename}_seg_{i}"

					label = ''.join(d["label"].splitlines())
					with open(f"{cell_file_name}.gt.txt", "w+") as f:
						f.write(label)

					cell_image = page.crop((x_min*5, y_min*5, x_max*5, y_max*5))
					cell_image.save(f"{cell_file_name}.png", dpi=(500, 500))
					# write data to txt
					# im = Image.fromarray(col_mask.astype(np.uint8),'L')
					

					col_x_mins[i%n_cols] = min(col_x_mins[i%n_cols], x_min)
					col_x_maxs[i%n_cols] = max(col_x_maxs[i%n_cols], x_max)
					col_y_mins[i%n_cols] = min(col_y_mins[i%n_cols], y_min)
					col_y_maxs[i%n_cols] = max(col_y_maxs[i%n_cols], y_max)

			if valid:
				im = Image.fromarray(col_mask.astype(np.uint8),'L')
				im.save(final_col_directory + filename + ".jpeg")
				# plt.show(im)

				im = Image.fromarray(table_mask.astype(np.uint8),'L')
				im.save(final_table_directory + filename + ".jpeg")

				im = Image.fromarray(cell_mask.astype(np.uint8),'L')
				im.save(final_cell_directory + filename + ".jpeg")
