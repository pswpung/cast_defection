FROM python:3.8.10

WORKDIR /cast_API

ADD . /cast_API

RUN pip install -r requirement.txt

CMD ["python", "server.py"]