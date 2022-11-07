FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN pip install scipy

RUN rm /etc/apt/sources.list.d/cuda.list
RUN sed -i "s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g" /etc/apt/sources.list \
    && rm -Rf /var/lib/apt/lists/* \
    && apt-get update --fix-missing -o Acquire::http::No-Cache=True \
    && apt-get install vim -y
RUN apt-get install wget -y 
RUN apt-get install libx11-dev libxext-dev libxtst-dev libxrender-dev libxmu-dev libxmuu-dev --fix-missing -y

# 启动ssh服务
RUN apt-get install openssh-server -y
RUN echo "Port 22">/etc/ssh/sshd_config && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
RUN echo "root:111111" | chpasswd
RUN mkdir /run/sshd
EXPOSE 22

COPY . /workspace/plate_chexing

WORKDIR /workspace/plate_chexing

CMD ["python","train.py"]
