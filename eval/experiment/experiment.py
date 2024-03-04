import numpy as np
from tqdm import tqdm
import pathlib

import torch
import torch.utils.data as data
from torch.optim import *

from sklearn.metrics import r2_score

from eval.model.model import CNNLSTMModel
from eval.experiment.dataset import GetDataset, collate_fn
from eval.experiment.utils import MOVIE_FPS
from eval.experiment.utils import RespType, RecordType, TrainTask, TrainType
from eval.experiment.utils import PearsonCorr

import os
import argparse

def Experiment(recordType, trainTask, trainType, useSecs):
    # Check if GPU is available (or CPU if not)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpuCount = os.cpu_count()
    print(device)

    seed = 42
    torch.manual_seed(seed)

    # Select training conditions
    respType = RespType.Thoracic
    recordType = RecordType(recordType)
    trainTask = TrainTask(trainTask)
    trainType = TrainType(trainType)
    # useSecs = 8
    batchSize = 64

    datasetDir = f'./dataset/'
    pathTmp = pathlib.Path(datasetDir)
    folders = [datasetDir + "/" + p.resolve().name + "/" for p in pathTmp.iterdir() if p.is_dir()]
    folders.sort()

    for targetNum in range(0, len(folders)):
        # Set datasets
        if trainType == TrainType.LeaveOneSubjectOut:
            num_epoch = 1
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
            num_epoch = len(folders)
            folderPath = folders[targetNum]
            dataset = GetDataset(folderPath=folderPath, recordType=recordType, respType=respType, trainTask=trainTask, useSecs=useSecs)
            samplesNum = len(dataset)
            test_size = int(samplesNum * 0.1)
            test_indices = list(range(samplesNum-test_size, samplesNum))
            train_size = int(samplesNum-test_size-MOVIE_FPS*useSecs)
            train_indices = list(range(0, train_size))
            trainDataset = torch.utils.data.dataset.Subset(dataset, train_indices)
            testDataset =  torch.utils.data.dataset.Subset(dataset, test_indices)

        # Set data loaders
        train_loader = data.DataLoader(
            trainDataset, batch_size=batchSize, shuffle=True,
            collate_fn=collate_fn, num_workers=cpuCount, pin_memory=True, drop_last=True,
        )
        test_loader = data.DataLoader(
            testDataset, batch_size=batchSize, shuffle=False,
            collate_fn=collate_fn, num_workers=cpuCount, pin_memory=True, drop_last=True,
        )

        # Set train params
        model = CNNLSTMModel()
        model.to(device)
        criterion = torch.nn.HuberLoss()
        learning_rate = 0.02
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        folderNames = [p.resolve().name for p in pathTmp.iterdir() if p.is_dir()]
        folderNames.sort()
        targetName = folderNames[targetNum]
        modelSavePath = f"./eval/model/state/model_{respType.name}_{recordType.name}_{trainTask.name}_{trainType.name}_{targetName}.pth"
        print(modelSavePath)
        
        # Training
        for epoch in range(num_epoch):
            model.train()
            scaler = torch.cuda.amp.GradScaler()
            for (inputs, labels) in tqdm(train_loader):
                inputs = inputs.to(device).float()
                labels = labels.to(device).float().unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                scaler.scale(loss).backward(retain_graph=True)
                scaler.step(optimizer)
                scaler.update()
        torch.save(model.state_dict(), modelSavePath)

        # Evaluation
        model.load_state_dict(torch.load(modelSavePath))
        model.eval() 
        with torch.no_grad():
            # Test
            labelResps = np.empty(0)
            outputResps = np.empty(0)
            for (inputs, labels) in tqdm(test_loader):
                inputs = inputs.to(device).float()
                labels = labels.to(device).unsqueeze(1).float()
                outputs = model(inputs)
                labelResps = np.append(labelResps, labels.to('cpu').detach().numpy().copy())
                outputResps = np.append(outputResps, outputs.to('cpu').detach().numpy().copy())

            if trainTask == TrainTask.Amplitude:
                corr = PearsonCorr(outputResps, labelResps)
                print("Corr: ", corr)
            elif trainTask == TrainTask.Gradient:
                r2 = r2_score(labelResps, outputResps)
                print("R2: ", r2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--recordType', choices=[i.value for i in RecordType], default=RecordType.Resting.value)
    parser.add_argument('--trainTask', choices=[i.value for i in TrainTask], default=TrainTask.Amplitude.value)
    parser.add_argument('--trainType', choices=[i.value for i in TrainType], default=TrainType.LeaveOneSubjectOut.value)
    parser.add_argument('--useSeconds', type=int, default=8)
    args = parser.parse_args()

    Experiment(args.recordType, args.trainTask, args.trainType, args.useSeconds)