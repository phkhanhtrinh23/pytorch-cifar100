FROM python:3.9

RUN mkdir /pytorch-cifar100-workspace
COPY . /pytorch-cifar100-workspace/

WORKDIR /pytorch-cifar100-workspace
RUN pip install -r requirements.txt

ENV HOME=/pytorch-cifar100-workspace

CMD [ "python" , "train.py", "--net", "vgg19", "--gpu" ]
