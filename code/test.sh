#!/bin/bash


DFILE=model
if [ ! -d "$DFILE" ]; then
    echo "$DFILE doesn't exist"
    wget $(python get_ya_url.py) -O model.zip
    unzip model.zip
fi

# docker
source activate solaris

# local
# source /home/brodt/kaggle/sbdr/solaris/bin/activate

testdatapath=$1
outputpath=$2
testdataargs="\
--testdir $testdatapath/SAR-Intensity \
--outputcsv $outputpath \
"

# docker
source settings.sh

# local
# source ./settings.sh

python baseline.py --pretest $testdataargs $settings

conda deactivate

# deactivate

sh ./test_models.sh $outputpath
