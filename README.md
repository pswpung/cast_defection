# Guide to mount volume
## 01. Download static Folder Here... 
https://drive.google.com/drive/folders/1wzNi4iJiFpQXZtckvVLrfhNMflsr0leH?usp=sharing
## 02. go to folder that contain staticfile in terminal 
## 03. run this commanad in terminal
docker run -v "$pwd/static:/cast_API/static" -d -p [stati local path]:5000 [container name]
