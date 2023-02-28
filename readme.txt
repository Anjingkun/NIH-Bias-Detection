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

Please make sure you have correctly installed aif360 package. If not, please install it manually. (AIF360 GitHub repository reference: https://github.com/Trusted-AI/AIF360)
