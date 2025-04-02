FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.
# RUN pip install tqdm
RUN pip install torch
RUN pip install transformers==4.48.3
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install numpy
