import pandas as pd
import os
import pathlib

from utils import RespData
from respShaping import respWaveShaping, respWaveGrad
from resizeFrames import resizeFrames


def findPath(dirName:str, dataType:RespData):
    targetPath = None
    if dataType == RespData.Thoracic:
        targetPath = dirName + '/resp_value_thoracic.tsv'
    if dataType == RespData.Abdominal:
        targetPath = dirName + '/resp_value_abdominal.tsv'
    if dataType == RespData.Movie:
        targetPath = dirName + '/RGB/'
    if dataType == RespData.MovieOutput:
        targetPath = dirName + '/RGB_Resize/'

    if dataType not in RespData:
        print("Datatype is not matched.")
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

    respAmpTho = respWaveShaping(filePath=findPath(dirName=dirPath, dataType=RespData.Thoracic), frameNum=frameNum, fmax=1.0)
    respAmpAbd = respWaveShaping(filePath=findPath(dirName=dirPath, dataType=RespData.Abdominal), frameNum=frameNum, fmax=1.0)

    respGradTho = respWaveGrad(respWave=respAmpTho)
    respGradAbd = respWaveGrad(respWave=respAmpAbd)

    respDataset = pd.DataFrame({'Thoracic' : pd.Series(respAmpTho).reset_index(drop=True),\
                                'Abdominal' : pd.Series(respAmpAbd).reset_index(drop=True)})
    respDataset.to_csv(outfilePath+"RespAmplitude.tsv", sep='\t')

    respGradDataset = pd.DataFrame({'Thoracic' : pd.Series(respGradTho).reset_index(drop=True),\
                                    'Abdominal' : pd.Series(respGradAbd).reset_index(drop=True)})
    respGradDataset.to_csv(outfilePath+"RespGradient.tsv", sep='\t')


def PreProcessing():
    dataPath = '../../data/'
    pathTmp = pathlib.Path(dataPath)
    folders = [dataPath + p.resolve().name + "/" for p in pathTmp.iterdir() if p.is_dir()]
    for folderPath in folders:
        print(folderPath)
        partTmp = pathlib.Path(folderPath)
        sessionDir = [folderPath + "/" + p.resolve().name + "/" for p in partTmp.iterdir() if p.is_dir()]
        for sessionPath in sessionDir:
            CreateDateSet(sessionPath, sessionPath)


if __name__ == "__main__":
    PreProcessing()