#!/bin/bash

# pytorch
# python mfnet_cmd.py --data test_data/data1.txt test_data/data2.txt --graph test_data/graph.edge_list --eval_locs test_data/eval.txt -o test_results/torch.out --type pytorch


# pyro mcmc
# python mfnet_cmd.py --data test_data/data1.txt test_data/data2.txt --graph test_data/graph.edge_list --eval_locs test_data/eval.txt -o test_results/pyro.out --type pyro --pyro_alg mcmc --num_samples 5 --burnin 5 --noisevar 1e-2


# pyro svi-iafflow
python mfnet_cmd.py --data test_data/data1.txt test_data/data2.txt --graph test_data/graph.edge_list --eval_locs test_data/eval.txt -o test_results/pyro.out --type pyro --pyro_alg svi-iafflow --num_samples 5 --noisevar 1e-2
