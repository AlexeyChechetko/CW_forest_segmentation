#!/bin/bash

URL=$(wget -qO- "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=${1}" | \
      grep -oP '"href":"\K[^"]+')

wget -O data.zip "$URL"

unzip data.zip

chmod -R u+rwx,go+x,o+x ./data