#!/bin/bash
apt-get install libsox-fmt-all libsox-dev sox > /dev/null
mkdir -p /content/data/converted /content/data/raw /content/modules
wget -q -O /content/data.zip https://solitabay.solita.fi/index.php/s/ZEbs8DcwbQqCWJA/download
unzip -j /content/data.zip "machine_spraak/raw_audio_data_20211007/*" -d /content/data/raw/
rm /content/data.zip
for file in /content/data/raw/*.WAV
do
	sox $file -r 48000 -b 32 /content/data/converted/`basename $file`
done
#wget -O /content/requirements.txt https://raw.githubusercontent.com/solita/ivves-machine-spraak/main/requirements.txt
wget -q -O /content/modules/utils.py https://raw.githubusercontent.com/solita/ivves-machine-spraak/main/modules/utils.py
#python3 -m pip install -r /content/requirements.txt
echo "Setup successful."