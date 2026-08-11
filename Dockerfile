FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
WORKDIR /simulus

RUN python -m pip install --upgrade pip

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY ./get_lpips.py .
RUN python get_lpips.py

RUN pip install craftax==1.4.5
