## Overview of cast_defection
 - API with Docker --> for creating Docker image
 - Train Script --> code for training model including EffNet, Xception and NASNetMobile

## How to train our Model
### 01. Download Train Script open Train Script_[model name].py example Train Script_EffNet.py
### 02. Download Dataset casting_data from kaggle (https://www.kaggle.com/ravirajsinh45/real-life-industrial-dataset-of-casting-product?select=casting_data) and locate in Train Script folder. 
The folder will look like...<br>
Train Script<br>
|- Train Script_EffNet.py<br>
|- Train Script_NasNetMobile.py<br>
|- Train Script_Xception.py<br>
|- casting_data<br>

### 03. change train_path and test_path to be your train and test path on your local computer
for example : <br>
train_path = "[your path]/Train Script/casting_data/casting_data/train/"<br>
test_path = "[your path]/Train Script/casting_data/casting_data/test/"
### 04. run the code with 
example : python "Train Script_EffNet.py"

<br>

## How to Create Docker Image...
### 01. Download API with Docker folder
### 02. run Dockerfile with this command...
docker image build [container name] .
### 03. mount static folder to our container

<br>

## How to mount volume??
### 01. Download static Folder Here... 
https://drive.google.com/drive/folders/1wzNi4iJiFpQXZtckvVLrfhNMflsr0leH?usp=sharing
### 02. go to folder that contain static folder in terminal 
### 03. run this commanad in terminal
docker run -v "$pwd/static:/cast_API/static" -d -p [stati local path]:5000 [container name]

<br>

## Endpoint API usage
01. Health Check <br>
<ul>
  <li>URL       : host:port/healthcheck</li>
  <li>Type      : GET</li>
  <li>Parameter : -</li>
</ul>

02. Predict <br>
<ul>
  <li>URL       : host:port/predict/[Model Name]</li>
  <li>Type      : POST</li>
  <li>Parameter : image (uploading image file)</li>
</ul>

full documentation --> https://drive.google.com/file/d/1tRiZyBkgIQODHYKSbnKkPhYR5_MA_6KC/view?usp=sharing
