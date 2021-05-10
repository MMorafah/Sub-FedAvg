#!/bin/bash

python main_u.py --dataset cifar10 --model lenet5 --ks 5 --in_ch 3 --nclass 2 --nsample_pc 250 --noniid --shard --num_users 100 --rounds 300 --frac 0.1 --local_bs 10 --local_ep 5 --lr 0.01 --momentum 0.5 --pruning_percent 10 --pruning_target 30 --dist_thresh 0.0001 --acc_thresh 50 --is_print