FROM nvcr.io/nvidia/tensorrt:20.09-py3
ENV HOME=/usr/src/app

# Install PyTorch
RUN pip install pip --upgrade
RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install onnx onnx-simplifier
RUN pip install optuna tqdm

WORKDIR $HOME
