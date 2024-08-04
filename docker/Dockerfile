FROM pytorch/pytorch:latest

RUN apt-get update
RUN pip install --upgrade pip

WORKDIR /app

COPY ./requirements.txt /app
COPY ./GUI.py /app
RUN pip install -r /app/requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["python", "GUI.py"]