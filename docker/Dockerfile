FROM python:3.8.10

WORKDIR /cast_API

COPY ["/requirement.txt" ,"/cast_API"]

RUN pip install -r requirement.txt

COPY ["/Cast-Defection Project/server.py", "/cast_API"]

CMD ["python", "server.py"]