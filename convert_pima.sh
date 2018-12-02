#!/bin/bash

cat data/pima.tsv |  awk 'BEGIN {FS="\\t";OFS=","} {$9 == 0 ? $9=";1,0" : $9=";0,1"; print}' | sed 's/,;/;/g' > data/pima.csv