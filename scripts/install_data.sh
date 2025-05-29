#!/bin/bash

mkdir data
cd data

YA_LINK="https://disk.yandex.ru/d/RdV0HxRmjCN4nw"

URL=$(wget -qO- "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=${YA_LINK}" | \
      grep -oP '"href":"\K[^"]+')

wget -O data.zip "$URL"

unzip data.zip

cd ..