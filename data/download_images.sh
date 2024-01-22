#!/usr/bin/env bash

cd $(dirname $0)

urlbase=`echo -n "aHR0cHM6Ly9kaWdpdGFsLnpsYi5kZS92aWV3ZXIvYXBpL3YxL3JlY29yZHMvCg==" | base64 -d`
cat imageurls.list | while read -r line; do image=${line#* }; ext=$(echo $image|grep -o "...$");  wget -O "$image" "${urlbase}${line% *}/files/images/${image/png/tif}/full/max/0/default.$ext"; done
