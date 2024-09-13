# ollama

```bash
#安装和更新
curl -fsSL https://ollama.com/install.sh | sh

ollama -v

ollama list


ollama pull llama3.1
ollama show llama3.1


ollama rm llama3.1

#server

./ollama serve

./ollama run llama3.1

#衣
pip install -U langchain langchain_experimental  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -qU langchain-ollama -i https://pypi.tuna.tsinghua.edu.cn/simple

# http://ai.baitech.com.cn:9901/

curl http://ai.baitech.com.cn:9901/api/embed -d '{
  "model": "all-minilm",
  "input": "Why is the sky blue?"}'

curl http://ai.baitech.com.cn:9901/generate -d '{
  "model": "llama3.1",
  "prompt":"Why is the sky blue?"
}'

curl http://ai.baitech.com.cn:9901/api/chat -d '{
  "model": "llama3.1",
  "messages": [
    { "role": "user", "content": "why is the sky blue?" }
  ]
}'

curl http://ai.baitech.com.cn:9901/api/tags

```

## 手工安装

```bash

wget https://ollama.com/install.sh -o install.sh
wget https://sourceforge.net/projects/ollama.mirror/files/v0.3.9/ollama-linux-amd64.tgz/download -o ollama-linux-amd64.tgz
wget https://sourceforge.net/projects/ollama.mirror/files/v0.3.9/ollama-linux-amd64-rocm.tgz/download -o ollama-linux-amd64-rocm.tgz

vi install.sh

sh install.sh

#sudo vi /etc/systemd/system/ollama.service
journalctl -u ollama -n 20

```