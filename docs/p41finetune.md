
# 大模型微调

## 参考资源

* <https://www.modelscope.cn>
* <https://github.com/hiyouga/LLaMA-Factory.git>
* <https://github.com/ggerganov/llama.cpp>
* <https://github.com/QwenLM/qwen.cpp>
* <https://www.modelscope.cn/models/qwen/qwen2-0.5b>
* <https://www.modelscope.cn/models/qwen/qwen2-7b-instruct>
* <https://www.modelscope.cn/datasets/AI-ModelScope/ruozhiba>

**论文**

* [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](papers/finetune/arxiv.org.pdf.2106.09685v2.pdf  ':ignore') 202108
* [Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](papers/finetune/arxiv.org.pdf.2303.15647v1.pdf ':ignore') 202305


## 练习

```bash
#install conda
#install git
sudo apt-get install git

conda create -p llm python=3.9
conda activate llm
#pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

#安装jupyter
conda install jupyterlab notebook --yes
#安装AI环境
conda install -c pytorch -c nvidia -c conda-forge tensorflow pytorch torchvision pytorch-cuda ultralytics
pip install  tensorflow-datasets tensorflow-text  -i https://pypi.tuna.tsinghua.edu.cn/simple

#langchain
pip install -U langchain langchain-community langchain_experimental  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install langchain-cli  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install python-dotenv  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install accelerate datasets oss2 addict  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install accelerate datasets oss2 addict  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install modelscope  -i https://pypi.tuna.tsinghua.edu.cn/simple


pip install ms-swift -u -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tf_keras -U -i https://pypi.tuna.tsinghua.edu.cn/simple

#下载模型  https://www.modelscope.cn
modelscope download --model qwen/Qwen2-7B-Instruct
ll /home/ubuntu/.cache/modelscope/hub/qwen

modelscope download --model qwen/qwen2-0.5b
ll ~/.cache/modelscope/hub/qwen/qwen2-0___5b/

#准备数据
git clone https://www.modelscope.cn/datasets/swift/self-cognition.git
#处理数据集


#下载 llama-factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

#启动界面
llamafactory-cli webui

#使用数据集对模型进行微调
#测试
#导出

#下载 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

python convert-hf-to-gguf.py /path/to/export

```