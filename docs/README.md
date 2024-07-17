# AI

## 学习资源

* Python <https://www.python.org/>
* Anaconda <https://www.anaconda.com/>
* Github <https://github.com/>
* Kaggle <https://www.kaggle.com/>
* Colab  <https://colab.research.google.com/>
* huggingface <http://www.huggingface.co>
* arxiv.org <https://arxiv.org/>

## 认识神经网络

* <https://caigk.github.io/convnetjs/demo/classify2d.html>
* <https://caigk.github.io/playground>

## 源码

* [deep-learning-from-scratch](https://github.com/oreilly-japan/deep-learning-from-scratch)
* [deep-learning-from-scratch-2](https://github.com/oreilly-japan/deep-learning-from-scratch-2)
* [deep-learning-from-scratch-4](https://github.com/oreilly-japan/deep-learning-from-scratch-4)
* [deep-learning-from-scratch-5](https://github.com/oreilly-japan/deep-learning-from-scratch-5)
* [deep-learning-with-python-notebooks](https://github.com/caigk/deep-learning-with-python-notebooks)

## 安装python环境

请先安装*Anaconda* https://www.anaconda.com/

* [yolo](https://docs.ultralytics.com/models/)
* [yolo world](https://docs.ultralytics.com/models/yolo-world/)
* [yolov8s-world.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-world.pt)

```bash
#创建并激活环境
conda create -n myenv python=3.11 --yes
conda activate myenv

#安装jupyter
conda install jupyterlab --yes
conda install pytorch torchvision -c pytorch --yes
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install openai-clip -i https://pypi.tuna.tsinghua.edu.cn/simple

#克隆 pytorch/tutorial.git
git clone git@github.com:pytorch/tutorials.git

#克隆 练习源码
git clone https://github.com/oreilly-japan/deep-learning-from-scratch.git
git clone https://github.com/caigk/deep-learning-with-python-notebooks.git

#使用yolo world 模型目标检测
#wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-world.pt
yolo yolov8s-world.pt source=/Users/caigangkun/Downloads/tls-talk640.jpg imgsz=640

## Install all packages together using conda
#conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics


```

## 安装IDE

* [VS CODE](https://code.visualstudio.com/)
* 安装扩展：ms-python.python，ms-python.debugpy

## 练习


## 加微信

![weixin](images/weixin.jpg)
