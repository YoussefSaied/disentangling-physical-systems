#!/bin/sh

bash ./move_models.sh

cd ../data/
bash ./move_datasets_py.sh
bash ./download_data.sh

cd ../../external/disentangling-vae/

python main.py btcvae_sdsc-latent-$1-beta-$2-alpha-$3 -d sdsc-day -l btcvae --lr 0.001 -b 1024 -e 30 -z $1 --btcvae-B $2 --btcvae-A $3
