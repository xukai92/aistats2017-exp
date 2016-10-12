#!/bin/bash
echo "total" > total.txt

for i in `seq 1 1`;
do
  pkill -Kill -f big-hmm
  bin/big-hmm -p 25 >> total.txt
  echo "" >> total.txt

  mv fork.csv fork$i.csv
done
