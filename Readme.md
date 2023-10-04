# Video Facial Asymmetry Analysis

This project provides a solution to analyze facial asymmetry in video files. The software reads videos from a specified directory, processes each frame to detect and align faces, and subsequently computes asymmetry metrics. It leverages a pre-trained model to make predictions and logs results for every video.

## Requirements
- Python 3.x
- Libraries: joblib, numpy, pandas, os, pytorch


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
python video_facial_asymmetry.py [path_to_videos_directory] --output_directory [path_to_output_directory]