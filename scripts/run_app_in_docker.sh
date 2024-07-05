#!/bin/bash

echo "Executando o IWSHAP.."
echo ""

python3 IWSHAP.py -s dataset/safe_dataset_reduced.csv -a dataset/attack_dataset_reduced_fabr.csv -x -n

echo "Processo finalizado!"

echo "==========================================================="
ls graphics/
echo "==========================================================="
ls logs/
echo "==========================================================="
ls reduced_data/
echo "==========================================================="

bash 