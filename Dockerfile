FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
#FROM --platform=linux/arm64 pytorch/pytorch:latest
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.
# RUN pip install tqdm
RUN pip install torch
RUN pip install transformers
RUN pip install scikit-learn
RUN pip install pandas
