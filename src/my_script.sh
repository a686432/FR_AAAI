#!/bin/sh
mkdir ../images/wg002
python main.py --gpu=0 --w=0.002 --outpath="../images/wg002" > "result/wg002.txt"
mkdir ../images/wg005
python main.py --gpu=0 --w=0.005 --outpath="../images/wg005" > "result/wg005.txt"



