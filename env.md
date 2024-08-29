```bash

conda create -n myenv python=3.11 --yes
conda activate myenv

#安装jupyter
conda install jupyterlab notebook --yes
#安装AI环境,tensorflow最好由CONDA安装
conda install tensorflow pytorch torchvision --yes
pip install labelme tensorflow-datasets tensorflow-text  -i https://pypi.tuna.tsinghua.edu.cn/simple

#langchain
pip install -U langchain langchain-community langchain_experimental  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install langchain-cli  -i https://pypi.tuna.tsinghua.edu.cn/simple

#
pip install python-dotenv  -i https://pypi.tuna.tsinghua.edu.cn/simple

#安装AI环境
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install openai-clip -i https://pypi.tuna.tsinghua.edu.cn/simple

```