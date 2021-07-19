## Cast Defection Classification
![python-badge](https://img.shields.io/badge/python->=3.8.10-blue?logo=python)
![tensorflow-badge](https://img.shields.io/badge/tensorfllow->=2.5.0-orange?logo=tensorflow)
![flask-badge](https://img.shields.io/badge/flask->=2.0.1-white?logo=flask)
### Train Model
> **Guide for training model**
01. Download [Cast-Defection Project](https://github.com/pswpung/cast_defection/tree/main/Cast-Defection%20Project)
02. Download Dataset casting_data from [kaggle dataset of cast product](https://www.kaggle.com/ravirajsinh45/real-life-industrial-dataset-of-casting-product?select=casting_data) and locate in Cast-Defection Project folder. <br>
The train folder will look like...
```
     Cast-Defection Project
            |- casting_data
            |    |- test
            |    |- train
            |
            |- train
            |    |- effnet
            |    |    |- main_effnet.py
            |    |    |- model_effnet.py
            |    |- nasnetmobile
            |    |    |- main_nasnetmobile.py
            |    |    |- model_nasnetmobile.py
            |    |- xception
            |    |    |- main_xception.py
            |    |    |- model_xception.py
            |    |- evaluation.py
            |
            |- server.py
```
03. run the main_[model name].py to train a model and save model to static folder autocatically.<br>
example code: 
```python
python main_effnet.py 
```
after training all model. Cast - Defection Project will look like this ...
```
     Cast-Defection Project
            |- casting_data
            |    |- test
            |    |- train
            |
            |- static
            |    |- trained model
            |         |-EffNet.h5
            |         |-NasNetMobile.h5
            |         |-Xception.h5
            |        
            |- train
            |    |- effnet
            |    |    |- main_effnet.py
            |    |    |- model_effnet.py
            |    |- nasnetmobile
            |    |    |- main_nasnetmobile.py
            |    |    |- model_nasnetmobile.py
            |    |- xception
            |    |    |- main_xception.py
            |    |    |- model_xception.py
            |    |- evaluation.py
            |
            |- server.py
```
> **Tips for training model.**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; you can adjust parameter without edit the python code by adding the configuration affter run "python main_[model name].py" like this "python main_[model_name].py [configuration]" (can add more than one configuration in the same time)
| parameter | default value | configurtion | note |
| ------------- | :---: | ------------- | ------------- |
| Number of Epochs | 10 | --n_epochs = [value] |  |
| Batch size | 4 | --train_batch_size = [value] |  |
| propotion for validation generater | 0.2 | --validation_split = [value] | [0<= validation_split <=1] |
| threshold value | 0.5 | --thresh = [value] | [0<= thresh <=1] |
| image size | 512 for Effnet and Xception <br> 224 for NasNetMobile | --input_size = [value] | NasNet mobile <= 224 only |

### Docker
> **Create Docker Container**
01. Download [Docker folder](https://github.com/pswpung/cast_defection/tree/main/docker)
02. run Dockerfile at cast_defection directory (parent directory)
```
docker image build -t [container name] -f docker/Dockerfile .
```
03. mount static folder to our container
> **mount volume**
01. Download [static Folder](https://drive.google.com/drive/folders/1wzNi4iJiFpQXZtckvVLrfhNMflsr0leH?usp=sharing) or use static folder that automatically create from train model
02. open terminal and run the following command
```
docker run -v "[cast_defection path]/cast_defection/Cast-Defection Project/static:cast_API/static" -d -p [local port]:5000 [container name]
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
<hr>

## Author
Pasawee Pungrasmi
