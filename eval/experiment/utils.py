import numpy as np
from enum import Enum

MOVIE_FPS = 30
INPUT_LENGTH = 10

class RespType(Enum):
    Thoracic = 1
    Abdominal = 2

class RecordType(Enum):
    Resting = 1,
    Dialogue = 2

class TrainTask(Enum):
    Amplitude = 1,
    Gradient = 2

class TrainType(Enum):
    LeaveOneSubjectOut = 1
    WithinSubject = 2

def PearsonCorr(outputs, targets):
    x = outputs
    y = targets
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    corr = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
    return corr