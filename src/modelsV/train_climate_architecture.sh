#!/bin/sh

bash ./move_models.sh

cd ../data/
bash ./move_datasets_py.sh
bash ./download_data.sh

cd ../../external/disentangling-vae/

python main.py btcvae_sdsc-climate-latent-$1 -d sdsc-original-shape -m Climate -l btcvae --lr 0.001 -b 256 -e 1 -z $1