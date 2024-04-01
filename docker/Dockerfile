FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update
RUN pip install --upgrade pip

WORKDIR /app

COPY ./requirements.txt /app
RUN pip install -r /app/requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:/app"
