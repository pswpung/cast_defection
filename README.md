# How to use this code...
## 01. Download cast_defection and unzip / pull cast_defection to your local computer
## 02. run Dockerfile with this command...
docker image build [container name]
## 03. mount static folder to our container 

# How to mount volume??
## 01. Download static Folder Here... 
https://drive.google.com/drive/folders/1wzNi4iJiFpQXZtckvVLrfhNMflsr0leH?usp=sharing
## 02. go to folder that contain static folder in terminal 
## 03. run this commanad in terminal
docker run -v "$pwd/static:/cast_API/static" -d -p [stati local path]:5000 [container name]
