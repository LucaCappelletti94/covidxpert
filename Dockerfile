FROM nvcr.io/nvidia/tensorflow:20.02-tf2-py3

EXPOSE 8888/tcp
WORKDIR /home
COPY . .
RUN apt-get update -qyy && apt-get install htop byobu python3-dev libasound2-dev -qyy
RUN pip install pip --upgrade && pip install ".[test]"
RUN pip install ipywidgets && jupyter nbextension enable --py widgetsnbextension