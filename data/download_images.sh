urlbase=`echo -n "aHR0cHM6Ly9kaWdpdGFsLnpsYi5kZS92aWV3ZXIvYXBpL3YxL3JlY29yZHMvCg==" | base64 -d`
cat imageurls.list | while read -r line; do wget -O "${line#* }" "${urlbase}${line% *}/files/images/${line#* }/full/max/0/default.tif"; done
