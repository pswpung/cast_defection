# Guide to mount volume
## 01. Download static Folder Here... 
https://drive.google.com/drive/folders/1wzNi4iJiFpQXZtckvVLrfhNMflsr0leH?usp=sharing
## 02. Move static folder to the same level with server.py
## 03. mount static folder to the container
docker run -v [local path]:/cast_API -d -p [stati local path]:5000 [container name]
