import numpy as np
import time
import cv2
import torch
import argparse

import threading
from multiprocessing import Manager, Process

from eval.model.model import CNNLSTMModel

LIST_SIZE = 300

def plotGraph(respArray):
    # Set graph window
    wname = "graph"
    graphPlotNum = int(500)
    graphHeight = int(250)
    graphWidth = int(800)
    graphSize = graphHeight, graphWidth, int(3)
    graphStartPosX = int(40)
    graphStartPosY = int(graphHeight / 5)
    graphEndPosX = int(graphWidth - 10)
    graphEndPosY = int(graphHeight * 4 / 5)
    graphNumberPosX = int(10);
    graphNumberAdjustPosY = int(5);

    while True:
        try:
            whiteImg = np.zeros(graphSize, dtype=np.uint8)
            whiteImg.fill(255)
            
            cv2.line(whiteImg, (graphStartPosX, graphEndPosY), (graphEndPosX, graphEndPosY), (0, 0, 0), 2)
            cv2.line(whiteImg, (graphStartPosX, graphStartPosY), (graphStartPosX, graphEndPosY), (0, 0, 0), 2)

            if len(respArray) > 1:
                plotResp = respArray[-graphPlotNum:]
                plotMaxValue = max(plotResp)
                plotMinValue = min(plotResp)
                if plotMaxValue - plotMinValue == 0:
                    plotMaxValue = 1.0
                    plotMinValue = -1.0
                drawPoints = []
                for i, value in enumerate(plotResp):
                    graphX = int(graphStartPosX + i * (graphEndPosX - graphStartPosX) / graphPlotNum)
                    graphY = int(graphStartPosY + (graphEndPosY - graphStartPosY) * (plotMaxValue - value) / (plotMaxValue - plotMinValue))
                    drawPoints.append([[graphX, graphY]])
                if len(drawPoints) > 0:
                    drawPoints = np.array(drawPoints).reshape((-1,1,2)).astype(np.int32)
                    cv2.polylines(whiteImg, [drawPoints], False, (0, 0, 255), 1)
                
                cv2.putText(whiteImg, "{:.0f}".format(plotMaxValue), (graphNumberPosX, graphStartPosY + graphNumberAdjustPosY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1);
                cv2.putText(whiteImg, "{:.0f}".format(plotMinValue), (graphNumberPosX, graphEndPosY + graphNumberAdjustPosY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1);

            cv2.imshow(wname, whiteImg)
            
            key = cv2.waitKey(1)&0xff
            if key == 27:
                break

        except KeyboardInterrupt:
            break

    return


def estThread(model, device, frames, respArray):   
    with torch.no_grad(): 
        input = torch.from_numpy(frames).to(device).float()
        input = input.unsqueeze(dim=0).transpose(2, 1)
        input = torch.cat([input, input], dim=0)
        output = model(input)
        value = output.to('cpu').detach().numpy().copy()
    
    respArray.append(value[0][0])
    if len(respArray) > LIST_SIZE:
        respArray.pop(0)
    
    return


def estimateResp(modelPath, respArray, useSeconds):
    # Set camera
    cap = cv2.VideoCapture(0)

    # Set estimate model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNLSTMModel()
    model.to(device)
    model.load_state_dict(torch.load(modelPath, map_location=device))
    model.eval()
    frameLength = int(useSeconds * 30)
    frameIntervalSec = useSeconds/10

    # Set frame window
    wname = "camera"
    windowSize = (540, 540)

    frame = None
    frameTime = None
    frames = np.empty(0)
    frameTimes = np.empty(0)
    frameAdjust = None
    th = None

    while True:
        try:
            # Get RGB frame
            _, frame = cap.read()
            if frame is not None:
                frameTime = time.time()
                if frameAdjust == None:
                    frame_width = len(frame[0])
                    frame_height = len(frame)
                    frameAdjust = int((frame_width - frame_height) / 2)
                if frameAdjust > 0:
                    frame = frame[:, frameAdjust:frameAdjust+frame_height]
                elif frameAdjust < 0:
                    frame = frame[-frameAdjust:-frameAdjust+frame_width, :]

            # Set input image
            if frame is not None:
                image = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
                B = image[:, :, 0]
                G = image[:, :, 1]
                R = image[:, :, 2]
                image = np.transpose(np.array([R, G, B]), [0, 1, 2])

                if len(frames) == 0:
                    frames = np.array([image])
                else:
                    frames = np.append(frames, [image], axis=0)
                frameTimes = np.append(frameTimes, frameTime)
                if len(frames) > frameLength:
                    frames = np.delete(frames, 0, 0)
                    frameTimes = np.delete(frameTimes, 0)
                
            # Estimate resp value
            if frameTime - frameTimes[0] > useSeconds:
                if th == None or th.is_alive() == False:
                    useFrames = [-len(frameTimes[frameTimes >= frameTime - i * frameIntervalSec]) for i in range(10)]
                    useFrames.reverse()
                    inputs = frames[useFrames]
                    th = threading.Thread(target=estThread, args=[model, device, inputs, respArray])
                    th.start()

            if frame is not None:
                frame = cv2.resize(frame, windowSize, interpolation=cv2.INTER_LINEAR)
                cv2.imshow(wname, frame)
            frame = None

            key = cv2.waitKey(1)&0xff
            if key == 27:
                break

        except KeyboardInterrupt:
            break

    if th is not None and th.is_alive():
        th.join()

    cap.release()
    cv2.destroyAllWindows()

    return


def main(modelPath, useSeconds):
    with Manager() as manager:
        respArray = manager.list()
        processEstResp = Process(target=estimateResp, args=[modelPath, respArray, useSeconds])
        processPlotGraph = Process(target=plotGraph, args=[respArray])

        processEstResp.start()
        processPlotGraph.start()
        while True:
            if processEstResp.is_alive() == False or processPlotGraph.is_alive() == False:
                break
        processEstResp.terminate()
        processPlotGraph.terminate()
        processEstResp.join()
        processPlotGraph.join()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--modelPath', type=str, default='./demo/pretrained/Amplitude_Resting_8sec.pth')
    parser.add_argument('--useSeconds', type=int, default=8)
    args = parser.parse_args()

    main(args.modelPath, args.useSeconds)
