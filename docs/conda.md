# anaconda

## 安装

* 官网：<https://www.anaconda.com>
* 清华镜像：<https://mirrors.ustc.edu.cn/anaconda/>

!> 在多用户在linux中，建议为每用户独立安装，环境初始化放在～/.bashr中

!> conda的配置文件为".condarc"，该文件在安装时不是缺省存在的。但是当你第一次运行conda config命令时它就被自动创建了。".condarc"配置文件遵循简单的YAML语法。

!> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

## 常用命令

```bash
conda --version

conda update conda

conda init

conda info

conda env list

conda create -n env-name python=3.11 numpy matplotlib
conda activate env-name

conda create -p ./conda python=3.11 numpy matplotlib
conda activate ./.conda


conda install nodebook 


conda deactivate

conda remove --name env_name --all

```

```bash
#获得环境中的所有配置
conda env export --name myenv > myenv.yml
#重新还原环境
conda env create -f  myenv.yml
```

