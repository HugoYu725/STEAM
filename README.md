# STEAM

Official implementation of the paper
**"Structural-Temporal Mining for Motif-Level Anomaly Detection in Dynamic Graphs"**

---

## 📄 Paper

https://www.sciencedirect.com/science/article/abs/pii/S095070512501007X

---

## 🔧 Environment

The core dependencies are as follows:

```bash
python==3.9
numpy==1.23.5
scipy==1.10.1
pandas==2.0.3
scikit-learn==1.3.1
matplotlib==3.7.2
networkx==3.1
tqdm==4.66.1
torch==1.12.0
torch_geometric==2.4.0
```

---

## ⚠️ PyTorch Geometric Dependencies

Please install the required PyG packages according to your CUDA version:

### CUDA 11.3

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
```

### CPU version

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
```

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Evaluate the model

```bash
python test.py
```

---

## 📊 Reproducibility

* Python: 3.9
* PyTorch: 1.12.0
* CUDA: 11.3

Please ensure that all dependencies are compatible.

---

## 📬 Contact

If you have any questions, feel free to open an issue or contact me.

---

## 🙏 Acknowledgement

If you find this work useful, please consider citing our paper.

---

## 📢 Note

We sincerely apologize for the delay in releasing the code. Due to a busy schedule recently, the implementation was not made available until now.  
The training code can be easily adapted from the testing code with minor modifications.  
Some additional datasets are not included in this repository due to their large size, but they can be easily obtained from public sources online.

非常抱歉，最近这段时间比较忙，拖到现在才更新代码。  
训练代码可以在测试代码的基础上进行简单修改即可。  
部分数据集由于体积较大未上传至本仓库，但可以在网络上方便地获取。
