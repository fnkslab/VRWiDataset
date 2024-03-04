import pandas as pd
import os
import pathlib

from eval.preprocess.utils import RespData
from eval.preprocess.respShaping import respWaveShaping, respWaveGrad
from eval.preprocess.resizeFrames import resizeFrames


def findPath(dirName:str, dataType:RespData):
    targetPath = None
    if dataType == RespData.Thoracic:
        targetPath = dirName + '/resp_value_thoracic.tsv'
    if dataType == RespData.Abdominal:
        targetPath = dirName + '/resp_value_abdominal.tsv'
    if dataType == RespData.Movie:
        targetPath = dirName + '/rgb-img/'

    if dataType not in RespData:
        print("Datatype is not matched.")
        return None
    
    return targetPath


def CreateDateSet(dirPath, outfilePath):
    # Check if not dirPath exists
    if not os.path.isdir(dirPath):
        print("The input directory does not exists.")
        return
    if not os.path.isdir(outfilePath):
        print("The output directory does not exists.")
        return

    resizeFrames(inputDir=findPath(dirName=dirPath, dataType=RespData.Movie), \
                 outputDir=findPath(dirName=outfilePath, dataType=RespData.Movie))
    
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
    dataPath = './data/'
    datasetPath = './dataset/'

    if not os.path.isdir(dataPath):
        print("Data directory does not exist. Please copy the directory from the server.")
        quit()
    os.makedirs(datasetPath, exist_ok=True)

    pathTmp = pathlib.Path(dataPath)
    folders = [p.resolve().name + "/" for p in pathTmp.iterdir() if p.is_dir()]
    folders.sort()
    for folderPath in folders:
        os.makedirs(datasetPath + folderPath, exist_ok=True)
        partTmp = pathlib.Path(dataPath + folderPath)

        sessionDir = [folderPath + "/" + p.resolve().name + "/" for p in partTmp.iterdir() if p.is_dir()]
        sessionDir.sort()
        for sessionPath in sessionDir:
            os.makedirs(datasetPath + sessionPath, exist_ok=True)
            CreateDateSet(dataPath + sessionPath, datasetPath + sessionPath)


if __name__ == "__main__":
    PreProcessing()
