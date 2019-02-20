#!/bin/bash

# DOWNLOAD THE DATASETS
mkdir -p dataset_zip
pushd dataset_zip
# TRAINING DATASET
if [[ $1 == all ]]; then
    echo "[INFO] Downloading train dataset"
    if [ ! -f clean_trainset_28spk_wav.zip ]; then
        wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip
    fi
    if [ ! -f noisy_trainset_28spk_wav.zip ]; then
        wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip
    fi
else
    echo "[INFO] Pass downloading train dataset, use 'all' argument for down all"
fi
# VALIDATION DATASET
echo "[INFO] Downloading valid dataset"
if [ ! -f clean_testset_wav.zip ]; then
    wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip
fi
if [ ! -f noisy_testset_wav.zip ]; then
    wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip
fi
popd

## INFLATE DATA
mkdir -p dataset_tmp
pushd dataset_tmp
if [[ $1 == all ]]; then
    echo "[INFO] Unzip train dataset"
    unzip -q -j ../dataset_zip/clean_trainset_28spk_wav.zip -d trainset_clean
    unzip -q -j ../dataset_zip/noisy_trainset_28spk_wav.zip -d trainset_noisy
fi
echo "[INFO] Unzip valid dataset"
unzip -q -j ../dataset_zip/clean_testset_wav.zip -d valset_clean
unzip -q -j ../dataset_zip/noisy_testset_wav.zip -d valset_noisy
popd

## RESAMPLE
if [[ $1 == all ]]; then
    declare -a arr=("trainset_clean" "trainset_noisy" "valset_clean" "valset_noisy")
else
    declare -a arr=("valset_clean" "valset_noisy")
fi
echo "[INFO] Resampling datasets: ${arr[*]}"
mkdir -p dataset
pushd dataset_tmp
for d in */; do
    mkdir -p "../dataset/$d"
    pushd "$d"
    for f in *.wav; do
        sox "$f" "../../dataset/$d$f" rate -v -I 16000
    done
    popd
done
popd

# REMOVE TMP DATA
rm -r dataset_tmp
