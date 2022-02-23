#!/bin/bash

pw=IhD999777
path_script1=/SceneNet/main.py
path_data1=/SceneNet/output
path_data2=/SceneNet/outputAORUS

sshpass -p $pw scp /home/david/BlenderProc$path_script1 david@192.168.1.2:/mnt/extern_hd$path_script1

if [ "$1" == data ]
then
sshpass -p $pw scp -r david@192.168.1.2:/mnt/extern_hd$path_data1 /home/david/BlenderProc$path_data2
fi
