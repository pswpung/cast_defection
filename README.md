## Cast Defection Classification
![python-badge](https://img.shields.io/badge/python->=3.8-blue?logo=python)
![tensorflow-badge](https://img.shields.io/badge/tensorfllow->=2.3-orange?logo=tensorflow)
![flask-badge](https://img.shields.io/badge/flask->=2.0-white?logo=flask)
### Train Model
01. Download [train folder](https://github.com/pswpung/cast_defection/tree/main/train)
02. Download Dataset casting_data from [kaggle dataset of cast product](https://www.kaggle.com/ravirajsinh45/real-life-industrial-dataset-of-casting-product?select=casting_data) and locate in train folder. <br>
The train folder will look like...
```
     train
       |- train_effnet.py
       |- train_nasnetmobile.py
       |- train_.py
       |- casting_data
```
03. change train_path and test_path to be your path on local computer. 
```
train_path = "[train folder path]/train/casting_data/casting_data/train/"
test_path = "[train folder path]/train/casting_data/casting_data/test/"
```
04. run the code below
```python
python train_effnet.py 
```

### Docker
> Create Docker Container
01. Download [Docker folder](https://github.com/pswpung/cast_defection/tree/main/Docker)
02. run Dockerfile
```
docker image build [container name] .
```
03. mount static folder to our container

> mount volume
5. Download [static Folder](https://drive.google.com/drive/folders/1wzNi4iJiFpQXZtckvVLrfhNMflsr0leH?usp=sharing)
6. open terminal and run this command
```
docker run -v "[static folder path]/static:cast_API/static" -d -p [local port]:5000 [container name]
```

### Endpoint API usage
```
01. Health Check 
         URL       : host:port/healthcheck
         Type      : GET
         Parameter : -
         Input     : -
         Output    : {This server is healthy}
         
02. Predict 
         URL       : http://host:port/predict/[Model Name]
         Type      : POST
         Parameter : image (uploading image file)
         Input     : { 
                       “image” : <image file1>,
                       “image” : <image file2>, 
                       “image” : <image file3}
                     }
         Output    : {
                       {image file1_name : { “Predict”: <image file1_predict>, “Probability”: <image file1_predict>}}, 
                       {image file2_name : { “Predict”: <image file2_predict>, “Probability”: <image file2_predict>}}, 
                       {image file3_name : { “Predict”: <image file3_predict>, “Probability”: <image file3_predict>}}
                     }

```
