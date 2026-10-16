from PIL import Image
from tqdm import tqdm
import cv2
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

def saveAllFrames(inputPath, outputDir, ext='jpg'):
    cap = cv2.VideoCapture(str(inputPath))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {inputPath}")

    output_path = pathlib.Path(outputDir)
    output_path.mkdir(parents=True, exist_ok=True)
    frame_num = 0

    try:
        for old_frame in output_path.glob(f"*.{ext}"):
            old_frame.unlink()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (RESIZE_FRAME_SIZE, RESIZE_FRAME_SIZE))
            frame_path = output_path / f"{frame_num:05d}.{ext}"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Failed to write frame: {frame_path}")
            frame_num += 1
    finally:
        cap.release()

    if frame_num == 0:
        raise RuntimeError(f"No video frames decoded: {inputPath}")
    return frame_num