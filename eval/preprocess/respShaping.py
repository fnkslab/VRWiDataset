import numpy as np
import pandas as pd
from scipy import signal, interpolate

from eval.preprocess.utils import MOVIE_FPS

def standalizeData(dataArray):
    mean = dataArray.mean()
    std = dataArray.std()
    result = (dataArray - mean) / std
    return result

def lowPassFilter(x, times, fmax):
    freq_X = np.fft.fftfreq(times.shape[0],times[1] - times[0])
    X_F = np.fft.fft(x)
    X_F[freq_X>fmax] = 0
    X_F[freq_X<-fmax] = 0
    x_CF  = np.fft.ifft(X_F).real
    return x_CF


def respWaveShaping(filePath, frameNum, fmax):
    dataFrame = pd.read_table(filePath)

    dataFrame = dataFrame.drop_duplicates(subset='DateTime')

    # remove wave trend
    dataFrame.RespValue = signal.detrend(dataFrame.RespValue)

    # 
    dateTime = pd.to_datetime(dataFrame.DateTime)
    
    waveStartTime = dateTime[0]
    elapsedTime = [(time - waveStartTime).total_seconds() for time in dateTime]
    record_func = interpolate.interp1d(elapsedTime, dataFrame.RespValue, fill_value = 'extrapolate', kind = 'cubic')

    # Resampling
    frameInterval = 1 / MOVIE_FPS
    resampleTime = np.array([frameInterval*num for num in range(frameNum)])
    resampleWave = record_func(resampleTime)

    # Low pass filter
    resampleWave = lowPassFilter(resampleWave, resampleTime, fmax)
    rowpass_func = interpolate.interp1d(resampleTime, resampleWave, fill_value = 'extrapolate', kind = 'cubic')
    
    shapedWave = rowpass_func(resampleTime)

    # Standalize wave
    shapedWave = standalizeData(shapedWave)

    return shapedWave


def CalculateGrad(respValueBefore, respValueAfter):
    respGrad = (respValueAfter - respValueBefore) / (2/30)
    return respGrad

def respWaveGrad(respWave):
    respGrad = np.array([CalculateGrad(respWave[i-1], respWave[i+1]) for i in range(1, respWave.size-1)], dtype=float)
    return np.append(None, respGrad, None)
