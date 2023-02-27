# NIH-Bias-Detection

- Team: Super2021
- Members: Yinghao Zhu, Jingkun An, Enshen Zhou, Hao Li, Haoran Feng

环境配置
OS				Python version
macOS			3.8 – 3.10
Ubuntu			3.8 – 3.10
Windows			3.8 – 3.10
(Optional) Create a virtual environment
Our tools mainly use AIF360 to provide service.And it requires specific versions of many Python packages which may conflict with other projects on your system. A virtual environment manager is strongly recommended to ensure dependencies may be installed safely. If you have trouble using our tools, try this first.

Install with pip
->conda create --name aif360 python=3.7
->conda activate aif360
->pip install aif360

Manual installation
->git clone https://github.com/Trusted-AI/AIF360
->pip install --editable '.[all]'