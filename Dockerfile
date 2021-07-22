FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN pip install --upgrade pip
RUN pip install transformers==4.8.2 flask==2.0.1

WORKDIR /app
COPY ./main.py .

EXPOSE 5000

CMD ["python", "main.py"]