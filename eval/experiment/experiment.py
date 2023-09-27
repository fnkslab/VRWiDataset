import numpy as np
from tqdm import tqdm
import pathlib

import torch
import torch.utils.data as data
from torch.optim import *

from sklearn.metrics import r2_score

from dataset import GetDataset, collate_fn
from model import CNNLSTMModel
from utils import MOVIE_FPS
from utils import RespType, RecordType, TrainTask, TrainType
from utils import PearsonCorr

import os

def Experiment():
    # Check if GPU is available (or CPU if not)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpuCount = os.cpuCount()
    print(device)

    seed = 42
    torch.manual_seed(seed)

    # Select training conditions
    respType = RespType.Thoracic
    recordType = RecordType.Resting
    trainTask = TrainTask.Amplitude
    useSecs = 8
    batchSize = 64
    modelSavePath = "./model_state/model.pth"

    trainType = TrainType.LeaveOneSubjectOut

    datasetDir = f'../../data/'
    pathTmp = pathlib.Path(datasetDir)
    folders = [datasetDir + "/" + p.resolve().name + "/" for p in pathTmp.iterdir() if p.is_dir()]
    folders.sort()
      
    for targetNum in range(0, 30):
        # Set datasets
        if trainType == TrainType.LeaveOneSubjectOut:
            firstTrainDataset = True
            testNums = [targetNum]
            for i, folderPath in enumerate(folders):
                if i in testNums:
                    testDataset = GetDataset(folderPath=folderPath, recordType=recordType, respType=respType, trainTask=trainTask, useSecs=useSecs)
                else:
                    if firstTrainDataset:
                        firstTrainDataset = False
                        trainDataset = GetDataset(folderPath=folderPath, recordType=recordType, respType=respType, trainTask=trainTask, useSecs=useSecs)
                    else:
                        addDataset = GetDataset(folderPath=folderPath, recordType=recordType, respType=respType, trainTask=trainTask, useSecs=useSecs)
                        trainDataset = data.ConcatDataset([trainDataset, addDataset])

        elif trainType == TrainType.WithinSubject:
            folderPath = folders[targetNum]
            dataset = GetDataset(folderPath=folderPath, recordType=recordType, respType=respType, trainTask=trainTask, useSecs=useSecs)
            samplesNum = len(testDataset)
            test_size = int(samplesNum * 0.1)
            test_indices = list(range(samplesNum-test_size, samplesNum))
            train_size = int(samplesNum-test_size-MOVIE_FPS*useSecs)
            train_indices = list(range(0, train_size))
            trainDataset = torch.utils.data.dataset.Subset(dataset, train_indices)
            testDataset =  torch.utils.data.dataset.Subset(dataset, test_indices)

        # Set data loaders
        train_loader = data.DataLoader(
            trainDataset, batch_size=batchSize, shuffle=True,
            collate_fn=collate_fn, num_workers=cpuCount, pin_memory=True,
        )
        test_loader = data.DataLoader(
            testDataset, batch_size=batchSize, shuffle=False,
            collate_fn=collate_fn, num_workers=cpuCount, pin_memory=True,
        )

        # Set train params
        model = CNNLSTMModel()
        model.to(device)
        criterion = torch.nn.HuberLoss()
        learning_rate = 0.02
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        # Training
        model.train()
        scaler = torch.cuda.amp.GradScaler()
        for (inputs, labels) in tqdm(train_loader):
            inputs = inputs.float().to(device)
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            scaler.scale(loss).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()
        torch.save(model.state_dict(), modelSavePath)

        # Evaluate model
        model.eval()
        with torch.no_grad():
            # Test
            labelResps = np.empty(0)
            outputResps = np.empty(0)
            for (images, labels) in tqdm(test_loader):
                inputs = inputs.float().to(device)
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                labelResps = np.append(labelResps, labels.to('cpu').detach().numpy().copy())
                outputResps = np.append(outputResps, outputs.to('cpu').detach().numpy().copy())

            if trainTask == TrainTask.Amplitude:
                corr = PearsonCorr(outputResps, labelResps)
                print("Corr: ", corr)
            elif trainTask == TrainTask.Gradient:
                r2 = r2_score(labelResps, outputResps)
                print("R2: ", r2)


if __name__ == "__main__":
    Experiment()