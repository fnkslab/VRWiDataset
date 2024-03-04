# VRWi Dataset

The VRWi dataset is designed for use in Video-Based Respiratory Waveform Estimation (VRWE) tasks. 

It comprises data from 30 individuals, evenly split between males and females. For each participant, the dataset includes two conditions: resting state and dialogue state. In each condition, there are face-masked video frames and respiratory waveform data from the thoracic/abdominal region. Additionally, for each face-masked video frame, the contour information of the image mask and the average RGB values within the mask are saved in RGB-FLD.txt. For the dialogue state, anonymized speech audio is also provided.


For details: [Video-Based Respiratory Waveform Estimation in Dialogue: A Novel Task and Dataset for Human-Machine Interaction](https://doi.org/10.1145/3577190.3614154)


## Directory structure
```
.
├── data
│   ├── F01-15, M01-15
│   │   └── session_resting, session_dialogue
│   │       ├── rgb-img
│   │       │   └── 00000.jpg-
│   │       ├── RGB-FLD.txt
│   │       ├── resp_value_thoracic.tsv
│   │       ├── resp_value_abdominal.tsv
│   │       └── audio.wav (only session_dialogue directory)
│   └── dataURLs.txt
├── eval
│   ├── experiment
│   │   ├── dataset.py
│   │   ├── experiment.py
│   │   └── utils.py
│   ├── preprocess
│   │   ├── preprocess.py
│   │   ├── resizeFrames.py
│   │   ├── respShaping.py
│   │   └── utils.py
│   └── model
│       ├── state
│       ├── convLSTM.py
│       └── model.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Data on VRWi Dataset
The data can be downloaded from the server.
```
$ cd data
$ wget -i dataURLs.txt
$ find ./ -type f -name "*.tgz" -exec tar -zxvf {} \;
```

| Data | Description |
| -- | -- |
| rgb-img/*.jpg | An image of a video frame with the face part masked for anonymization |
| RGB-FLD.txt | A text file containing output information obtained using [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format) and the average RGB values of the masked facial area |
| resp_value_thoracic.tsv | A TSV file recording sensor values obtained from a respiratory sensor belt worn on the thorax, along with the recording timestamps |
| resp_value_abdominal.tsv | A TSV file recording sensor values obtained from a respiratory sensor belt worn on the abdomen, along with the recording timestamps |
| audio.wav | Audio data anonymized by cutting high-frequency components with 2*F0 |

## Run Experiments
1. Clone this repository.
```
$ git clone https://github.com/fnkslab/VRWiDataset
$ cd VRWiDataset
```

2. Download data from the server.
```
$ cd data
$ wget -i urlData.txt
$ find ./ -type f -name "*.tgz" -exec tar -zxvf {} \;
$ cd ../
```

3. Prepare environment for Python 3.10 and install requirements.
```
$ pip install -r requirements.txt
```

4. Preprocess data
```
$ python -m eval.preprocess.preprocess
```

5. Run experiments
```
$ python -m eval.experiment.experiment \
    --recordType Rest \
    --trainTask Amp \
    --trainType LOSO \
    --useSeconds 8
```


## License
This dataset is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License](https://creativecommons.org/licenses/by-nc/4.0/), unless otherwise noted.
The source code in the 'eval' folder is licensed MIT.


## Citation
```
@inproceedings{10.1145/3577190.3614154,
author = {Obi, Takao and Funakoshi, Kotaro},
title = {Video-Based Respiratory Waveform Estimation in Dialogue: A Novel Task and Dataset for Human-Machine Interaction},
year = {2023},
isbn = {9798400700552},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3577190.3614154},
doi = {10.1145/3577190.3614154},
booktitle = {Proceedings of the 25th International Conference on Multimodal Interaction},
pages = {649–660},
numpages = {12},
series = {ICMI '23}
}
```
