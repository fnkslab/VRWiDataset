import numpy as np
import pandas as pd
import cv2
import torch
import datetime
import argparse

from eval.model.model import CNNLSTMModel

LIST_SIZE = 300


def main(modelPath, videoPath, useSeconds):
    # Set movie
    movie = cv2.VideoCapture(videoPath)
    if movie.isOpened() == False:
        print("Cannot open movie file.")
        return
    movie.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Get movie propaties
    width = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = movie.get(cv2.CAP_PROP_FPS)
    frameAdjust = int((width - height) / 2)
    
    # Set estimate model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNLSTMModel()
    model.to(device)
    model.load_state_dict(torch.load(modelPath, map_location=device))
    model.eval()

    frameInterval = int(useSeconds * fps/10)
    frames = np.empty(0)
    frameCount = 0
    frameTime = datetime.timedelta(seconds=1/fps)
    dist = pd.DataFrame(columns=['Time', 'EstimateResp'])
    
    while True:
        _ret, frame = movie.read()
        if _ret == False:
            break
        if frameAdjust > 0:
            frame = frame[:, frameAdjust:frameAdjust+height]
        elif frameAdjust < 0:
            frame = frame[-frameAdjust:-frameAdjust+width, :]
        
        frame = cv2.resize(frame, (256,256), interpolation=cv2.INTER_LINEAR)
        B = frame[:, :, 0]
        G = frame[:, :, 1]
        R = frame[:, :, 2]
        frame = np.transpose(np.array([R, G, B]), [0, 1, 2])
    
        if len(frames) == 0:
            frames = np.array([frame])
        else:
            frames = np.append(frames, [frame], axis=0)
        if len(frames) > LIST_SIZE:
            frames = np.delete(frames, 0, 0)
        
        if len(frames) > (frameInterval*10):
            inputs = frames[-(frameInterval*10)::frameInterval]
            input = torch.from_numpy(inputs).to(device).float()
            input = input.unsqueeze(dim=0).transpose(2, 1)
            input = torch.cat([input, input], dim=0)
            with torch.no_grad():
                output = model(input)
            value_tho = output.to('cpu').detach().numpy().copy()
            time_now = frameTime*frameCount
            dist = dist.append({'Time': time_now, 'EstimateResp': value_tho[0][0]}, ignore_index=True)
                   
        frameCount += 1

    dist.to_csv(f"./EstimateResp.csv")
    movie.release()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--modelPath', type=str, default='./demo/pretrained/Amplitude_Resting_8sec.pth')
    parser.add_argument('--videoPath', type=str, default='')
    parser.add_argument('--useSeconds', type=int, default=8)
    args = parser.parse_args()

    main(args.modelPath, args.videoPath, args.useSeconds)
