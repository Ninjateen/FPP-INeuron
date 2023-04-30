FROM python:3.11-slim-buster
WORKDIR /app
COPY . /app/
RUN python3 -m pip install -r requirements.txt
