# SSVEPLite 🧠⚡

[English](#english) | [中文](#chinese)

<a name="english"></a>
## English Version

A minimalist convolutional neural network for SSVEP classification, adapted from the EEGNet architecture.

## 📋 Overview

This repository presents a simplified 2-layer CNN architecture for Steady-State Visual Evoked Potential (SSVEP) classification. The model, modified from the original EEGNet architecture, retains only the essential temporal and spatial convolution layers while achieving competitive performance on the Nakanishi et al. (2015) dataset.

## 🏆 Key Results

Performance comparison on Nakanishi et al. (2015) dataset (1.0s window):

| Model | Accuracy (%) | ITR (bits/min) | Evaluate Method |
|------|------------|----------------|----------|
| **SSVEPLite**（Our Work） | **94.35 ± 6.71** | **188.02 ± 29.63** | within-subject |
| SSVEPformer | 77.18±26.63 | 137.30±72.87 | within-subject |

⚠️ **Important Information **: The above results are based on a within-subject evaluation. The cross-subject (LOSO) results are to be added due to a bug in the original codebase where the LOSO branch was commented out.

## 🏗️ Architecture

The SSVEPLite architecture consists of only two convolutional layers:
1. **Temporal Convolution**: Extracts frequency-domain features from EEG time series
2. **Spatial Convolution**: Learns spatial filters across EEG channels

This design is inspired by EEGNet but simplified to its core components, removing depthwise separable convolutions, Average pool layers, and dropout layers.

## 💻 Implementation Details

### Model Structure
```python
SSVEPLite(
  (temporal_conv): nn.Conv2d(1, F1, (1, kernelength), bias=False, padding='same')
  (spatial_conv): Conv2dWithConstraint(F1, F1*D, (num_channels, 1), bias=False, groups=F1, max_norm=1.)
  (classifier): LinearWithConstraint(F2*T, num_classes, max_norm=0.25)
)
```

## 📊 Dataset

**Nakanishi et al. (2015)**
- Subjects: 10
- Classes: 12
- Channels: 8
- Sampling rate: 256 Hz
- Trial duration: 4s (1s window used for classification)

## 🚀 Usage

### Requirements
```
python>=3.7
pytorch>=1.8.0
numpy==1.22.3
pandas==1.4.3
torch==1.10.1
scipy==1.9.0
matplotlib==3.5.2
einops==0.8.0
pyyaml==6.0.1
```

### Training & Evaluation
```bash
cd Test
python Classifier_Test.py
```

## 📁 Repository Structure
```
SSVEP-Nakanishi2015-MinimalCNN/
├── Model/
│   ├── SSVEPLite.py          # Simplified CNN architecture
│   ├── Other models from the original repo
├── Train/
│   ├── Classifier_Train.py   # Training functions (added ITR calculation)
│   └── Trainer_Script.py     # Added new model configuration
├── Test/
│   ├── Classifier_Test.py    # Fixed commented code from original repo
├── Utils/
│   ├── Ploter.py             # Added ITR logging
│   ├── Other utilities from the original repo
├── etc/
│   ├── config.yaml           # Added new model parameters
│   ├── Other configs from the original repo
├── data/
│   ├── Dial/
│   │   ├── DataSub_1.mat
│   │   ...
│   │   ├── LabSub_1.mat
├── Resource/
│   ├── requirements.txt
└── README.md
```

## 🙏 Acknowledgments

This implementation builds upon:
- YuDong Pan's [DL_Classifier repository](https://github.com/YuDongPan/DL_Classifier), which provided the SSVEPformer implementation and evaluation framework
- The EEGNet architecture by Lawhern et al., which inspired the temporal-spatial convolution design

## 📚 Citation

If you use this code in your research, please cite the original works:

[1] Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013. https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

[2] J. Chen, Y. Zhang, Y. Pan, P. Xu, and C. Guan, “A transformer-based deep neural network model for SSVEP classification,” Neural Networks, May 2023, doi: https://doi.org/10.1016/j.neunet.2023.04.045.

## 📄 License

This project is licensed under the MIT License.

---

<a name="chinese"></a>
## 中文版本

一个用于SSVEP分类的极简卷积神经网络，基于EEGNet架构改进。

### 📋 概述

本项目提出了一个简化的2层CNN架构，用于稳态视觉诱发电位（SSVEP）分类。该模型基于原始EEGNet架构修改而来，仅保留了最核心的时间卷积和空间卷积层，在Nakanishi等人（2015）数据集上取得了优异的性能。

### 🏆 主要结果

在Nakanishi等人（2015）数据集上的性能对比（1.0秒时间窗）：

| 模型 | 准确率 (%) | ITR (bits/min) | 评估方式 |
|------|------------|----------------|----------|
| **SSVEPLite**（本工作） | **94.35 ± 6.71** | **188.02 ± 29.63** | 被试内 |
| SSVEPformer | 77.18±26.63 | 137.30±72.87 | 被试内 |

⚠️ **重要说明**：以上结果基于被试内（within-subject）评估。由于原始代码库中LOSO分支被注释掉的bug，跨被试（LOSO）结果待补充。

### 🏗️ 架构设计

SSVEPLite架构仅包含两个卷积层：
1. **时间卷积**：从脑电时间序列中提取频域特征
2. **空间卷积**：学习跨脑电通道的空间滤波器

该设计受EEGNet启发，但简化到了核心组件，移除了深度可分离卷积、池化层和dropout层。

### 💻 实现细节

#### 模型结构
```python
SSVEPLite(
  (temporal_conv): nn.Conv2d(1, F1, (1, kernelength), bias=False, padding='same')
  (spatial_conv): Conv2dWithConstraint(F1, F1*D, (num_channels, 1), bias=False, groups=F1, max_norm=1.)
  (classifier): LinearWithConstraint(F2*T, num_classes, max_norm=0.25)
)
```

### 📊 数据集

**Nakanishi等人（2015）**
- 受试者：10人
- 类别：12类
- 通道数：8
- 采样率：256 Hz
- 试验时长：4秒（使用1秒窗口进行分类）

### 🚀 使用方法

#### 环境要求
```
python>=3.7
pytorch>=1.8.0
numpy==1.22.3
pandas==1.4.3
torch==1.10.1
scipy==1.9.0
matplotlib==3.5.2
einops==0.8.0
pyyaml==6.0.1
```

#### 训练模型&评估性能
```bash
cd Test
python Classifier_Test.py
```

### 📁 项目结构
```
SSVEP-Nakanishi2015-MinimalCNN/
├── Model/
│   ├── SSVEPLite.py          # 简化的CNN架构
│   ├── 来自原始仓库的其它模型
├── Train/
│   ├── Classifier_Train.py   # 训练函数（增加ITR计算）
│   └── Trainer_Script.py     # 增加新模型设置
├── Test/
│   ├── Classifier_Test.py    # 修复原始仓库注释
├── Utils/
│   ├── Ploter.py   # 增加ITR记录
|   ├── 来自原始仓库的其它函数
├── etc/
│   ├── config.yaml   # 增加新模型参数
|   ├── 来自原始仓库的其它函数
├── data/
│   ├── Dial/
|   |   ├── DataSub_1.mat
         ...
|   |   ├── LabSub_1.mat
├── Resource/
│   ├── requirements.txt
└── README.md
```

### 🙏 致谢

本实现基于以下工作：
- YuDong Pan的[DL_Classifier仓库](https://github.com/YuDongPan/DL_Classifier)，提供了SSVEPformer实现和评估框架
- Lawhern等人的EEGNet架构，启发了时空卷积设计

### 📚 引用

如果您在研究中使用了本代码，请引用以下原始工作：

[1] Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013. https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

[2] J. Chen, Y. Zhang, Y. Pan, P. Xu, and C. Guan, “A transformer-based deep neural network model for SSVEP classification,” Neural Networks, May 2023, doi: https://doi.org/10.1016/j.neunet.2023.04.045.
‌

### 📄 许可证

本项目采用MIT许可证。