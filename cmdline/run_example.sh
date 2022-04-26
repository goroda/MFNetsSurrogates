#!/bin/bash

# pytorch
# python mfnet_cmd.py --data example_input_files/data1.txt example_input_files/data2.txt --graph example_input_files/graph.edge_list --eval_locs example_input_files/eval.txt -o example_results_files/torch.out --type pytorch


# pyro mcmc
# python mfnet_cmd.py --data example_input_files/data1.txt example_input_files/data2.txt --graph example_input_files/graph.edge_list --eval_locs example_input_files/eval.txt -o mcmc_example_results_files/pyro.out --type pyro --pyro_alg mcmc --num_samples 5000 --burnin 1000 --noisevar 1e-2


# pyro svi-iafflow
python mfnet_cmd.py --data example_input_files/data1.txt example_input_files/data2.txt --graph example_input_files/graph.edge_list --eval_locs example_input_files/eval.txt -o iafflow_example_results_files/pyro.out --type pyro --pyro_alg svi-iafflow --num_samples 5000 --num_steps 2000 --iaf_depth 1 --noisevar 1e-2

# pyro svi-multinormal
# python mfnet_cmd.py --data example_input_files/data1.txt example_input_files/data2.txt --graph example_input_files/graph.edge_list --eval_locs example_input_files/eval.txt -o multinormal_example_results_files/pyro.out --type pyro --pyro_alg svi-multinormal --num_samples 5000 --num_steps 2000 --noisevar 1e-2


# postprocess
# python mfnet_cmd_postprocess.py mcmc_example_results_files
# python mfnet_cmd_postprocess.py iafflow_example_results_files 
# python mfnet_cmd_postprocess.py multinormal_example_results_files && mv posterior.pdf multinormal_example_results_files/
