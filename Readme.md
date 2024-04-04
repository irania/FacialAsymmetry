[# Video Facial Asymmetry Analysis

This project provides a solution to analyze facial asymmetry in videos of smiling individuals and predict the chance of Parkinson's disease. The software reads videos from a specified directory, processes each frame to detect and align faces, and subsequently computes asymmetry metrics. It leverages a pre-trained model from the UT-Parkinson dataset to make predictions and logs the results for each video.


## Requirements
- Python 3.x
- Libraries: joblib, numpy, pandas, os, pytorch
- Videos should be in the `.mp4` format.


## Setup

### Create Venv
- python -m venv ./venv
- .\venv\Scripts\activate

### Install Reuirements
- pip install -r requirements.txt

### Install Spiga
- install pytorch base on the cuda version of the computer
- git clone https://github.com/andresprados/SPIGA.git
- cd spiga
- pip install -e .

### Extract OpenFace2.2.0 from below link and put it beside other root folders

link: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Windows-Installation

## Usage
python video_facial_asymmetry.py \[path_to_videos_directory\] --output_directory \[path_to_output_directory\]](Readme.md)