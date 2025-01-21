import glob
import os

import numpy as np
from PIL import Image

from utils import check_and_remove_white_background, is_background_white

root = "/data/Coding/ImageClusteringDEXA/output_rmbk/"
output_path = "output_bkremoved"
os.makedirs(output_path, exist_ok=True)

files = glob.glob(root + "/*_or.png")
for file in files:
    image = np.asarray(Image.open(file).convert("L"))
    image = check_and_remove_white_background(image)
    output_file = os.path.join(
        output_path, os.path.basename(file).replace("_or.png", "_rm.png")
    )
    Image.fromarray(image).save(output_file)
    print()
