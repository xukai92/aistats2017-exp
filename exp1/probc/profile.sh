#!/bin/bash
echo "total" > total.txt

for i in `seq 1 10`;
do
  bin/big-hmm -p 150 >> total.txt
  echo "," >> total.txt

  mv fork.csv fork$i.csv
done
