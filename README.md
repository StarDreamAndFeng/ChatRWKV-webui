> 在 Google Colab 运行

# 从 Hugging Face 下载模型

-  安装依赖

```python
!pip install diffusers==0.4.0
```

-  登录 HF

```python
from huggingface_hub import notebook_login
notebook_login()
```

-  模型名称-下载仓库-保存位置

```python
GOOGLE_DRIVE_PATH = "/content/ChatRWKV-webui/models" #@param {type:"string"}
REPO_ID = "BlinkDL/rwkv-4-pile-7b" #@param {type:"string"}
FILE_NAME = "RWKV-4-Pile-7B-EngChn-testNovel-done-ctx2048-20230317.pth" #@param {type:"string"}
```

# 克隆本项目

-  下载项目

```bash
!git clone https://github.com/StarDreamAndFeng/ChatRWKV-webui.git
```

- 创建 models 目录

用于保存模型

```bash
%cd ChatRWKV-webui/
%mkdir models
%cd models
%pwd
```

将之前下载的模型复制到 models 目录下

```bash
%cp $filePath /content/ChatRWKV-webui/models
```

删除之前下载的模型目录,清理空间

```bash
%rm -rf /content/hrcache
```

# 安装必要的依赖

```python
!pip install numpy tokenizers prompt_toolkit
!pip install torch --extra-index-url https://download.pytorch.org/whl/cu117 --upgrade
!pip install gradio
!pip install rwkv
!pip install ninja
```

# 运行

```python
!python ChatRWKV-webui/app.py
```



