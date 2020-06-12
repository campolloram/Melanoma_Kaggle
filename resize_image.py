import os
import glob
from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed

def resize_image(image_path, output_folder,resize):
	base_name = os.path.basename(image_path)
	outpath = os.path.join(output_folder, base_name)
	img = Image.open(image_path)
	img = img.resize(
		(resize[0],resize[1]), resample = Image.BILINEAR
		)
	img.save(outpath)

if __name__ == '__main__':
	# Train set
	ImageFile.LOAD_TRUNCATED_IMAGES = True
	input_folder = '/home/pollo/Documents/Kaggle/melanoma_project/data/jpeg/train'
	output_folder = '/home/pollo/Documents/Kaggle/melanoma_project/data/images/train224'

	images = glob.glob(os.path.join(input_folder, '*.jpg'))
	Parallel(n_jobs = 12)(
		delayed(resize_image)(
			i,
			output_folder,
			(224,224)
		) for i in tqdm(images)
		)

	
	# Test set
	input_folder = '/home/pollo/Documents/Kaggle/melanoma_project/data/jpeg/test'
	output_folder = '/home/pollo/Documents/Kaggle/melanoma_project/data/images/test224'


	images = glob.glob(os.path.join(input_folder, '*.jpg'))
	Parallel(n_jobs = 12)(
		delayed(resize_image)(
			i,
			output_folder,
			(224,224)
		) for i in tqdm(images)
		)