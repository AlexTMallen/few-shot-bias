#!/usr/bin/env bash
mkdir -p data
wget https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip -O winogrande.zip
unzip -d data/tmp winogrande.zip
mv data/tmp/winogrande_1.1 data/winogrande
rm -r data/tmp winogrande.zip
