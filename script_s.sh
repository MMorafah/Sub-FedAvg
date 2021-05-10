#!/bin/bash

python main_s.py --dataset cifar10 --model lenet5 --ks 5 --in_ch 3 --nclass 2 --nsample_pc 250 --noniid --shard --num_users 100 --rounds 300 --frac 0.1 --local_bs 10 --local_ep 5 --lr 0.01 --momentum 0.5 --pruning_percent_ch 0.45 --pruning_percent_fc 10 --pruning_target 50 --dist_thresh_ch 0.01 --dist_thresh_fc 0.0005 --acc_thresh 50 --is_print