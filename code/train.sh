#!/bin/bash

# docker
source activate solaris

cp ./src/core.py /opt/conda/envs/solaris/lib/python3.7/site-packages/solaris/utils/
cp ./src/geo.py /opt/conda/envs/solaris/lib/python3.7/site-packages/solaris/utils/

# local
# source /home/brodt/kaggle/sbdr/solaris/bin/activate


traindatapath=$1

traindataargs="\
--sardir $traindatapath/SAR-Intensity \
--opticaldir $traindatapath/PS-RGB \
--labeldir $traindatapath/geojson_buildings \
"

# docker
source settings.sh

# local
# source ./settings.sh

python baseline.py --pretrain $traindataargs $settings

conda deactivate

sh ./train_models.sh
