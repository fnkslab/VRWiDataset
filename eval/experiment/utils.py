import numpy as np
from enum import Enum

MOVIE_FPS = 30
INPUT_LENGTH = 10

class RespType(Enum):
    Thoracic = "Tho"
    Abdominal = "Abd"

class RecordType(Enum):
    Resting = "Rest",
    Dialogue = "Dial"

class TrainTask(Enum):
    Amplitude = "Amp",
    Gradient = "Grad"

class TrainType(Enum):
    LeaveOneSubjectOut = "LOSO"
    WithinSubject = "Within"

def PearsonCorr(outputs, targets):
    x = outputs
    y = targets
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    corr = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
    return corr