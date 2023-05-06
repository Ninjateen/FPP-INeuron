FROM python:3.8-slim-buster
WORKDIR /app
COPY . /app/
RUN apt update -y
RUN pip install dill
RUN python3 -m pip install -r requirements.txt
CMD ["python3", "app.py"]
