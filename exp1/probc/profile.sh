#!/bin/bash
echo "total" > total.txt

for i in `seq 1 10`;
do
  pkill -Kill -f big-hmm
  bin/big-hmm -p 30 >> total.txt
  echo "" >> total.txt

  mv fork.csv fork$i.csv
done
