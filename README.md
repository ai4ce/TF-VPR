# Self-Supervised Visual Place Recognition by Mining Temporal and Feature Neighborhoods
[Chao Chen](https://scholar.google.com/citations?hl=en&user=WOBQbwQAAAAJ), [Xinhao Liu](https://gaaaavin.github.io), [Xuchu Xu](https://www.xuchuxu.com), [Li Ding](linkedin.com/in/li-ding-056251ab), [Yiming Li](https://scholar.google.com/citations?user=i_aajNoAAAAJ), [Ruoyu Wang](https://github.com/ruoyuwangeel4930), [Chen Feng](https://scholar.google.com/citations?user=YeG8ZM0AAAAJ)

**"A Novel self-supervised VPR model capable of retrieving positives from various orientations."**

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
[![Linux](https://svgshare.com/i/Zhy.svg)](https://svgshare.com/i/Zhy.svg)
[![GitLab issues total](https://badgen.net/github/issues/ai4ce/V2X-Sim)](https://github.com/Joechencc/TF-VPR)
[![GitHub stars](https://img.shields.io/github/stars/ai4ce/V2X-Sim.svg?style=social&label=Star&maxAge=2592000)](https://github.com/Joechencc/TF-VPR/stargazers/)
<div align="center">
    <img src="https://s2.loli.net/2022/07/30/ZldqmQGFhajCxRn.png" height="300">
</div>
<br>

# PointNetVlad-Pytorch
Unofficial PyTorch implementation of PointNetVlad (https://github.com/mikacuy/pointnetvlad)

I kept almost everything not related to tensorflow as the original implementation.
The main differences are:
* Multi-GPU support
* Configuration file (config.py)
* Evaluation on the eval dataset after every epochs

This implementation achieved an average top 1% recall on oxford baseline of 84.81%

### Pre-Requisites
* PyTorch 0.4.0
* tensorboardX

### Generate pickle files
```
cd generating_queries/

# For training tuples in our baseline network
python generate_training_tuples_cc_baseline_batch.py

# For network evaluation
python generate_test_cc_sets.py
```

### Train
```
python train_pointnetvlad.py --dataset_folder $DATASET_FOLDER
```

### Evaluate
```
python evaluate.py --dataset_folder $DATASET_FOLDER
```

Take a look at train_pointnetvlad.py and evaluate.py for more parameters
