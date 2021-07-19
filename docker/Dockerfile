FROM python:3.8.10

WORKDIR /cast_API

COPY ["/requirement.txt", "/cast_defection_project/server.py" ,"/cast_API/"]

RUN pip install -r requirement.txt

CMD ["python", "server.py"]