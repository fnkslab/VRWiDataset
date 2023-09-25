import re
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
import pathlib
import datetime
from tqdm import tqdm
import tarfile

from utils import RespData
from respShaping import respWaveShaping, respWaveGrad
from resizeFrames import resizeFrames


def findPath(dirName:str, dataType:RespData):
    pathTmp = pathlib.Path(dirName)
    targetPath = None
    if dataType == RespData.Thoracic:
        targetPath = list(pathTmp.glob('**/*resp_value_Port1*.tsv'))[0].resolve()
        print(targetPath)
    if dataType == RespData.Abdominal:
        targetPath = list(pathTmp.glob('**/*resp_value_Port2*.tsv'))[0].resolve()
        print(targetPath)
    if dataType == RespData.Movie:
        targetPath = dirName + '/RGB/'
    if dataType == RespData.MovieOutput:
        targetPath = dirName + '/RGB_Resize/'

    if dataType not in RespData:
        print("Datatype is not matched.")
        return None
    if targetPath == None:
        print("The file could not be found.")
        return None
    return targetPath


def CreateDateSet(dirPath, outfilePath):
    # Check if not dirPath exists
    if not os.path.isdir(dirPath):
        print("The input directory cound not be found.")
        return
    if not os.path.isdir(outfilePath):
        print("The output directory cound not be found.")
        return

    resizeFrames(inputDir=findPath(dirName=dirPath, dataType=RespData.Movie), \
                 outputDir=findPath(dirName=dirPath, dataType=RespData.MovieOutput))
    
    pathTmp = pathlib.Path(findPath(dirName=dirPath, dataType=RespData.Movie))
    frameNum = len([p.resolve().name for p in pathTmp.iterdir()])


    respWaveTho = respWaveShaping(filePath=findPath(dirName=dirPath, dataType=RespData.Thoracic), frameNum=frameNum, fmax=1.0)
    respWaveAbd = respWaveShaping(filePath=findPath(dirName=dirPath, dataType=RespData.Abdominal), frameNum=frameNum, fmax=1.0)

    respGradTho = respWaveGrad(respWave=respWaveTho)
    respGradAbd = respWaveGrad(respWave=respWaveAbd)

    respDataset = pd.DataFrame({'Thoracic' : pd.Series(respWaveTho).reset_index(drop=True),\
                                'Abdominal' : pd.Series(respWaveAbd).reset_index(drop=True)})
    respDataset.to_csv(outfilePath+"RespAmplitude.tsv", sep='\t')

    respGradDataset = pd.DataFrame({'Thoracic' : pd.Series(respGradTho).reset_index(drop=True),\
                                    'Abdominal' : pd.Series(respGradAbd).reset_index(drop=True)})
    respGradDataset.to_csv(outfilePath+"RespGradient.tsv", sep='\t')
