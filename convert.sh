#!/bin/zsh
# Usage: ./convert.sh <input folder> <output folder>
# Convert all wav files from input folder to a common format (48 kHz, 24 bit) and
# store them in the output folder
setopt local_options extended_glob
for file in $1/*(#i).wav(.);
do
	echo "Converting $file";
	sox $file -r 48000 -b 24 $2/`basename $file`;
done
