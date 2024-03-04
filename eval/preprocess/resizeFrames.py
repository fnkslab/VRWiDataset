from PIL import Image
from tqdm import tqdm
import os
import pathlib

from eval.preprocess.utils import RESIZE_FRAME_SIZE

def resizeImg(inputPath, outputPath):
    img = Image.open(inputPath)
    img_resize = img.resize((RESIZE_FRAME_SIZE, RESIZE_FRAME_SIZE))
    img_resize.save(outputPath)

def resizeFrames(inputDir, outputDir):
    if not os.path.isdir(inputDir):
        print("The directory cound not be found.")
        return
    os.makedirs(outputDir, exist_ok=True)
    pathTmp = pathlib.Path(inputDir)
    images = [p.resolve().name for p in pathTmp.iterdir()]
    images.sort()
    for imageName in tqdm(images):
        if imageName == '.gitkeep':
            continue
        resizeImg(inputDir+imageName, outputDir+imageName)
