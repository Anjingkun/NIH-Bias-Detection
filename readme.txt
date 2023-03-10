# NIH-Bias-Detection

- Team: Super2021
- Members: Yinghao Zhu, Jingkun An, Enshen Zhou, Hao Li, Haoran Feng

## Usage

- `measure_disparity.py`: detect and evaluate the bias (prediction logits dependent)
- `mitigate_disparity.py`: mitigate bias
- `example_adult.ipynb`: an example tutorial that measures and mitigates disparity on adult census income dataset (AdultDataset)
- `example_meps.ipynb`: an example tutorial that measures and mitigates disparity on clinical dataset (MEPSDataset19)

## Environment Setup

Linux/Windows/MacOS with Python version >= 3.8

(We've tested on Ubuntu 18 and Debian 11)

(Optional) Create a virtual environment with conda

- Install with pip

```
conda create -n befair python=3.9
conda activate befair
pip install -r requirements.txt
```

Please follow the instructions in `datasets/README.md` to download the datasets. An easier way is to download from GitHub Releases and put them in `datasets/data/raw` folder

Folder structure:

```bash
datasets/
  ├── data/
  │   └── raw/
  │       ├── adult/
  │       │   ├── adult.data
  │       │   ├── adult.names
  │       │   ├── adult.test
  │       │   └── README.md
  │       ├── meps/
  │       │   ├── h181.csv
  │       │   ├── h192.csv
  │       │   ├── generate_data.R
  │       │   └── README.md
  │       └── ...
  └── utils/
measure_disparity.py
mitigate_disparity.py
example_adult.ipynb
example_meps.ipynb
```
