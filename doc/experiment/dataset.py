import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import re
import pathlib

import torch
import torch.utils.data as data

from utils import MOVIE_FPS, INPUT_LENGTH
from utils import RespType, RecordType, TrainTask

class Dataset(data.Dataset):
    def __init__(self, dirPath:str, respType:RespType, trainTask:TrainTask, useSecs:int):
        super().__init__()

        self.RGBPath = [str(p) for p in sorted(Path(dirPath).glob("**/RGB_Resize/*.jpg"))]

        if trainTask == TrainTask.Amplitude:
            df_respData = pd.read_csv(dirPath + 'RespAmplitude.tsv', sep='\t').reset_index(drop=True)
        elif trainTask == TrainTask.Gradient:
            df_respData = pd.read_csv(dirPath + 'RespGradient.tsv', sep='\t').reset_index(drop=True)
        if respType == RespType.Thoracic:
            self.respData = pd.concat([self.respData, df_respData.Thoracic])
        elif respType == RespType.Abdominal:
            self.respData = pd.concat([self.respData, df_respData.Abdominal])
        self.respData = self.respData.reset_index(drop=True)

        self.len = len(self.respData) - MOVIE_FPS*useSecs - 1

        self.interval = MOVIE_FPS/INPUT_LENGTH*useSecs
        self.useSecs = useSecs

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        ps = self.RGBPath[index:index+MOVIE_FPS*self.useSecs:self.interval]
        images = []
        for p in ps:
            img = Image.open(p)
            img = np.array(img)
            img = np.transpose(img, (2, 0, 1))
            images.append(img)
        imageSeq = np.stack(images, axis=1)
        imageSeq = torch.from_numpy(imageSeq)

        respValue = self.respData[index+(INPUT_LENGTH-1)*self.interval]

        return imageSeq, respValue


def GetDataset(folderPath:str, recordType:RecordType, respType:RespType, trainTask:TrainTask, useSecs:int) -> Dataset:
    num_temp = pathlib.Path(folderPath)
    if recordType == RecordType.Resting:
        sessionDir = [folderPath + "/" + p.resolve().name + "/" for p in num_temp.iterdir() if p.is_dir() and re.match('session_resting*', p.resolve().name)]
    elif recordType == RecordType.Dialogue:
        sessionDir = [folderPath + "/" + p.resolve().name + "/" for p in num_temp.iterdir() if p.is_dir() and re.match('session_dialogue*', p.resolve().name)]
    for i, recordDir in enumerate(sessionDir):
        if i == 0:
            dataset = Dataset(dirPath=recordDir, respType=respType, trainTask=trainTask, useSecs=useSecs)
        else:
            addDataset = Dataset(dirPath=recordDir, respType=respType, trainTask=trainTask, useSecs=useSecs)
            dataset = data.ConcatDataset([dataset, addDataset])
    return dataset


def collate_fn(batch):
    imageSeqs, respValues = [], []

    for imageSeq, diff in batch:
        if diff == diff:
            if not np.isnan(diff):
                imageSeqs.append(imageSeq)
                respValues.append(diff)
            else:
                print("remove not float data")
        else:
            print("remove NaN data")
    
    imageSeqs = torch.stack(imageSeqs, dim=0)
    respValues = torch.tensor(respValues)

    return imageSeqs, respValues