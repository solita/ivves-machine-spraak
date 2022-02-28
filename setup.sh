#!/bin/bash
apt-get install libsox-fmt-all libsox-dev sox > /dev/null
mkdir -p /content/data/converted /content/data/raw /content/modules
wget -q -O /content/data.zip "https://solitabay.solita.fi/index.php/s/ZEbs8DcwbQqCWJA/download?path=%2F$1"
unzip -j /content/data.zip "$1/*" -d /content/data/raw/
rm /content/data.zip
for file in /content/data/raw/*.WAV
do
	sox $file -r 48000 -b 32 -e floating-point /content/data/converted/`basename $file`
done
#wget -O /content/requirements.txt https://raw.githubusercontent.com/solita/ivves-machine-spraak/main/requirements.txt
# wget -q -O /content/modules/utils.py https://raw.githubusercontent.com/solita/ivves-machine-spraak/main/modules/utils.py
wget -q -O /content/modules/utils.py https://raw.githubusercontent.com/solita/ivves-machine-spraak/test-solitabay-folders/modules/utils.py
# wget -q -O /content/modules/pca_clustering.py https://raw.githubusercontent.com/solita/ivves-machine-spraak/main/modules/pca_clustering.py
wget -q -O /content/modules/pca_clustering.py https://raw.githubusercontent.com/solita/ivves-machine-spraak/test-solitabay-folders/modules/pca_clustering.py
#python3 -m pip install -r /content/requirements.txt
echo "Setup successful."
