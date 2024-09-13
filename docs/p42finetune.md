
# 大模型微调

## 一、参考资源

* [DB-GPT 框架](https://github.com/eosphoros-ai/DB-GPT)
* [Text2SQL 微调](https://github.com/eosphoros-ai/DB-GPT-Hub)
* [DB-GPT-WEB](https://github.com/eosphoros-ai/DB-GPT-Web)
* [Awesome-Text2SQL](https://github.com/eosphoros-ai/Awesome-Text2SQL)
* [第三方评估GPT-4](https://www.numbersstation.ai/post/nsql-llama-2-7b)
* [训练的权重文件](https://huggingface.co/eosphoros)

* [xlang-ai](https://github.com/xlang-ai)

* [llama-cpp-python](https://llama-cpp-python.readthedocs.io/en/latest/)

**数据集**

* [SPIDER](https://yale-lily.github.io/spider)
* [WikiSQL](https://github.com/salesforce/WikiSQL)
* [CHASE](https://xjtu-intsoft.github.io/chase/)
* [BIRD-SQL](https://bird-bench.github.io/)
* [CoSQL](https://yale-lily.github.io/cosql)
* [NSQL](https://github.com/NumbersStationAI/NSQL)
* [20W dataset](https://huggingface.co/datasets/Healthy13/Text2SQL/tree/main)


## 二、transformers

```bash

python src/transformers/models/llama/convert_llama_weights_to_hf.py \
   --input_dir /home/ubuntu/caigk/Meta-Llama-3-8B-Instruct --model_size 7B --output_dir /output/path

```


## 三、练习 llama-recipes


```bash

git clone git@github.com:meta-llama/llama-recipes.git
cd llama-recipes

conda create -p .conda python=3.11
conda activate ./.conda

pip install llama-recipes
pip install llama-recipes[vllm]
pip install llama-recipes[auditnlg]

#下载模型
pip install modelscope
modelscope download --model LLM-Research/Meta-Llama-3-8B-Instruct --local_dir ../Meta-Llama-3-8B-Instruct
#https://www.modelscope.cn/models/llm-research/meta-llama-3.1-8b-instruct
modelscope download --model llm-research/meta-llama-3.1-8b-instruct --local_dir ../meta-llama-3.1-8b-instruct

#数据集

#Llama3 中文化数据集  https://www.modelscope.cn/datasets/baicai003/Llama3-Chinese-dataset
git clone https://www.modelscope.cn/datasets/baicai003/Llama3-Chinese-dataset.git ../Llama3-Chinese-dataset
#shareAI-Llama3 中文化偏好数据集 https://www.modelscope.cn/datasets/shareAI/shareAI-Llama3-DPO-zh-en-emoji
git clone https://www.modelscope.cn/datasets/shareAI/shareAI-Llama3-DPO-zh-en-emoji.git ../shareAI-Llama3-DPO-zh-en-emoji
```



## 四、练习 DBGPT

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


#

conda create -p .conda python=3.11
conda activate ./.conda

pip install -e ".[default]"
cp .env.template  .env

conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit cuda-nvcc -y --copy
CMAKE_ARGS="-DLLAVA_BUILD=OFF" pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy==1.26.4
#pip install -e ".[llama_cpp]"

pip install llama-cpp-python[server]
pip install openai

#取模型
mkdir models
cd models
git clone https://www.modelscope.cn/thomas/text2vec-large-chinese.git
git clone https://www.modelscope.cn/qwen/Qwen2-7B-Instruct.git
git clone https://www.modelscope.cn/ZhipuAI/glm-4-9b-chat.git


pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
modelscope download --model thomas/text2vec-large-chinese --local_dir models/text2vec-large-chinese
modelscope download --model ZhipuAI/glm-4-9b-chat --local_dir models/glm-4-9b-chat


bash ./scripts/examples/load_examples.sh

#启动服务
dbgpt start webserver --port 6006
#LLM_MODEL_PATH=/home/ubuntu/caigk/DB-GPT-main/models/Qwen2-7B-Instruct    python dbgpt/app/dbgpt_server.py --port 6006


pip install dashscope


```

```python
#https://llama-cpp-python.readthedocs.io/en/latest/
from llama_cpp import Llama
llm = Llama(
      model_path="../export/Export-7.6B-F16.gguf",
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)
output = llm(
      "Q: Name the planets in the solar system? A: ", # Prompt
      max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(output)

```

## 五、练习 llama2-finetune


```bash

git clone git@github.com:FangxuY/llama2-finetune.git
cd llama2-finetune

conda create -p .conda python=3.11
conda activate ./.conda





```


## firefly

* https://github.com/yangjianxin1/Firefly