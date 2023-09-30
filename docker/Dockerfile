FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV RAY_RESULTS_PATH=/home/ray/project/ray_results
ENV PYTHONWARNINGS="ignore::DeprecationWarning"

RUN pip install --upgrade pip setuptools
COPY ./requirements.txt ./requirements.txt
RUN python -m pip install -r requirements.txt

RUN git config --global --add safe.directory /home/ray

# RUN sudo chmod 0777 /home/ray/project

# LaTeX for matplotlib
# RUN apt-get install texlive-binaries texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super -y
# RUN pip install latex
