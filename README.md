# VRWi Dataset

The VRWi dataset is designed for use in Video-Based Respiratory Waveform Estimation (VRWE) tasks.

It comprises data from 30 individuals, evenly split between males and females. For each participant, the dataset includes two conditions: resting state and dialogue state. In each condition, there are face-masked videos and respiratory waveform data from the thoracic/abdominal region. Additionally, for each face-masked video frame, the contour information of the image mask and the average RGB values within the mask are saved in RGB-FLD.txt. For the dialogue state, anonymized speech audio is also provided.

The recordings are now distributed as MP4 files through **[GitHub Releases](https://github.com/fnkslab/VRWiDataset/releases/tag/data-mp4)**. (Update October 2026)

For details: [Video-Based Respiratory Waveform Estimation in Dialogue: A Novel Task and Dataset for Human-Machine Interaction](https://doi.org/10.1145/3577190.3614154)

## Directory structure
```
.
├── data
│   ├── F01-15, M01-15
│   │   └── session_resting, session_dialogue(*)
│   │       ├── video.mp4
│   │       ├── RGB-FLD.txt
│   │       ├── resp_value_thoracic.tsv
│   │       ├── resp_value_abdominal.tsv
│   │       └── audio.wav (dialogue sessions only)
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
├── demo
│   ├── pretrained
│   │   ├── Amplitude_Resting_8sec.pth
│   │   ├── Amplitude_Dialogue_7sec.pth
│   │   ├── Gradient_Resting_4sec.pth
│   │   └── Gradient_Dialogue_4sec.pth
│   ├── camera.py
│   └── video.py
├── .gitignore
├── README.md
└── requirements.txt
```
(*) F03, F07, and M07 have two separate dialogue sessions, stored as `session_dialogue_1` and `session_dialogue_2` instead of `session_dialogue`.

## Data on VRWi Dataset
The data can be downloaded from GitHub Releases.
```
$ cd data
$ wget -i dataURLs.txt
$ find ./ -type f -name "*.tgz" -exec tar -zxvf {} \;
```

| Data | Description |
| -- | -- |
| video.mp4 | A recording video with the face part masked for anonymization |
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

2. Download data from GitHub Releases.
```
$ cd data
$ wget -i dataURLs.txt
$ find ./ -type f -name "*.tgz" -exec tar -zxvf {} \;
$ cd ../
```

3. Prepare an environment for Python 3.10 and install requirements.
```
$ pip install -r requirements.txt
```

4. Preprocess data. This extracts and resizes MP4 frames into `dataset/<participant>/<session>/rgb-img/`, and generates the respiratory waveform targets.
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

## Demo

### Pretrained models
Each model was trained separately on either the full resting or dialogue session data.
Every weight file follows the naming convention `(EstimateTarget)_(RecordType)_(InputSeconds).pth` (e.g. `Amplitude_Resting_8sec.pth`).

### Run demo scripts
1. Real-time estimation with a webcam.
```
$ python -m demo.camera \
    --modelPath ./demo/pretrained/Amplitude_Resting_8sec.pth \
    --useSeconds 8
```

2. Estimation on a pre-recorded video.
```
$ python -m demo.video \
    --modelPath ./demo/pretrained/Amplitude_Resting_8sec.pth \
    --videoPath ./path/to/video.mp4 \
    --useSeconds 8
```

## License
This dataset is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License](https://creativecommons.org/licenses/by-nc/4.0/), unless otherwise noted.
The source code in the 'eval' and 'demo' folder is licensed MIT.

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
