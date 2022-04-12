#!/bin/bash

pw=IhD999777
ip=192.168.1.4
client=/home/david/BlenderProc/SceneNet
server=/mnt/extern_hd/SceneNet
script1=/main.py
script2=/Helper.py
blend=/objects
data_client=/outputExtern
data_server=/output

sshpass -p $pw scp $client$script1 david@$ip:$server$script1
sshpass -p $pw scp $client$script2 david@$ip:$server$script2
sshpass -p $pw scp -r $client$blend david@$ip:$server


if [ "$1" == data ]
then
sshpass -p $pw scp -r david@$ip:$server$data_server $client$data_client
fi
