# Nasz obraz będzie dzidziczył z obrazu Ubuntu w wersji latest
FROM ubuntu:latest

# Instalujemy niezbędne zależności. Zwróć uwagę na flagę "-y" (assume yes)
RUN apt update && apt install -y figlet

WORKDIR /app

RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN python3 -m pip install pandas
RUN python3 -m pip install numpy
RUN python3 -m pip install torch
RUN python3 -m pip install torchvision

COPY ./zadanie1.py ./
COPY ./Customers.csv ./
COPY ./train.py ./

RUN chmod +r ./Customers.csv
RUN chmod +x ./train.py

ARG epochs=5
RUN echo $epochs

RUN python3 ./train.py $epochs