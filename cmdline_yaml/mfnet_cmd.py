"""Command Line Utility for MFNETS."""
import sys
# sys.path.append("../mfnets_surrogates")
sys.path.append("/Users/alex/Software/mfnets_surrogate/mfnets_surrogates")
import yaml
import os

from collections import namedtuple

import argparse
import networkx as nx
import torch
import numpy as np
import matplotlib.pyplot as plt

import net_torch as net
import net_pyro

from pyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide import (
    # AutoDelta,
    AutoNormal,
    AutoMultivariateNormal,
    # AutoLowRankMultivariateNormal,
    # AutoGuideList,
    AutoIAFNormal,
    # init_to_feasible,
)


import pandas as pd

import logging
logging.basicConfig(level=logging.INFO)

ModelTrainData = namedtuple('ModelTrainData', ('train_in', 'train_out', 'dim_in', 'dim_out', 'output_dir'))

def fill_graph(graph, input_spec, model_info):
    """Assign node and edge functions."""

    if input_spec['graph']['connection_type'] == 'scale-shift':
        logging.info('Pointwise scale-shift graph is used')
        for node in graph.nodes:

            # works because model names must match in the input file and in the graph.edge_list file
            dim_in = model_info[node].dim_in
            dim_out = model_info[node].dim_out
            logging.info(f"Updating function for graph node {node}: dim_in = {dim_in}, dim_out = {dim_out}")
            graph.nodes[node]['func'] = torch.nn.Linear(dim_in, dim_out, bias=True)
            graph.nodes[node]['dim_in'] = dim_in
            graph.nodes[node]['dim_out'] = dim_out

        for e1, e2 in graph.edges:

            # rho needs to multiply output of lower fidelity model and be of the dimension of output of high-fidelity model
            dim_in = model_info[e2].dim_in
            dim_out_rows = model_info[e2].dim_out
            dim_out_cols = model_info[e1].dim_out
            logging.info(f"Updating function for graph edge {e1}->{e2} (rho_[e1->e2](x)): dim_in = {dim_in}, dim_out = {dim_out_rows} x {dim_out_cols}, but flattened")
            graph.edges[e1, e2]['func'] = torch.nn.Linear(dim_in, dim_out_rows * dim_out_cols, bias=True)
            graph.edges[e1, e2]['out_rows'] = dim_out_rows
            graph.edges[e1, e2]['out_cols'] = dim_out_cols
            graph.edges[e1, e2]['dim_in'] = dim_in
    else:
        logging.error(f"Connection type {model_info['graph']['connection_type']} is not recognized")
        exit(1)

    return graph

def parse_graph(input_spec, model_info):
    """Parse the graph."""
    graph_file = input_spec['graph']['structure']

    try:
        with open(graph_file) as f:
            edge_list = f.read().splitlines()
    except FileNotFoundError:
        print(f"Cannot open file {graph_file}")
        exit(1)

    graph = fill_graph(nx.parse_edgelist(edge_list, create_using=nx.DiGraph), input_spec, model_info)

    roots = set([x for x in graph.nodes() if graph.in_degree(x) == 0])

    logging.info(f"Root models: {roots}")
    
    return graph, roots

def parse_model_info(args):
    """Parse data files."""
    num_models = input_spec['num_models']
    logging.info(f"Number of models: {num_models}")

    models = {}
    for model in input_spec['model_info']:

        name = str(model['name'])

        try: 
            train_input = pd.read_csv(model['train_input'], delim_whitespace=True)
        except FileNotFoundError:
            print(f"Cannot open training inputs for model {name} in file {model['train_input']}")
            exit(1)

        try: 
            train_output = pd.read_csv(model['train_output'], delim_whitespace=True)
        except FileNotFoundError:
            print(f"Cannot open training outputs for model {name} in file {model['train_output']}")
            exit(1)            
            
        assert train_input.shape[0] == train_output.shape[0]

        dim_in = train_input.shape[1]
        dim_out = train_output.shape[1]

        output_dir = os.path.join(os.getcwd(), model['output_dir'])
        
        models[name] = ModelTrainData(train_input, train_output, dim_in, dim_out, output_dir)
        logging.info(f"Model {name}: number of inputs = {dim_in}, number of outputs = {dim_out}, ntrain = {train_output.shape[0]}, output_dir = {output_dir}")

    return models

def parse_evaluation_locations(input_spec):
    """Parse eval_locations."""
    model_evals = {} 
    for ii, model in enumerate(input_spec['model_info']):

        name = str(model['name'])

        if 'test_output' in model:
            filename = model['test_output']
            logging.info(f"Will evaluate model {name} at inputs of file {filename}")

            try: 
                test_input = pd.read_csv(filename, delim_whitespace=True)
            except FileNotFoundError:
                print(f"Cannot open test inputs for model {name} in file {filename}")
                exit(1)
                
            model_evals[name] = test_input
        else:
            model_evals[name] = None
            
    return model_evals

def model_info_to_dataloaders(model_info, graph_nodes):
    """Convert datasets to dataloaders for pytorch training."""
    data_loaders = []
    for node in graph.nodes:
        model = model_info[node]
        # print(model)
        x = model.train_in.to_numpy()
        if x.ndim == 1:
            x = x[:, np.newaxis]
        y = model.train_out.to_numpy()
        if y.ndim == 1:
            y = y[:, np.newaxis] 

        dataset = net.ArrayDataset(torch.Tensor(x), torch.Tensor(y))
        data_loaders.append(torch.utils.data.DataLoader(dataset, batch_size=x.shape[0], shuffle=False))

    return data_loaders

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='mfnet_cmd',
        description="Perform MFNETS",
    )

    parser.add_argument('input_file',  type=str, nargs=1, help='YAML input file')

    #########################
    ## Parse Arguments
    args = parser.parse_args()

    input_file = args.input_file[0]
    logging.info(f"Reading input Specs: {input_file}")
    try:
        with open(input_file, 'r') as file:
            input_spec = yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Cannot open input file {input_file}")
        exit(1)


    print(input_spec)
    
    model_info = parse_model_info(input_spec)

    graph, roots = parse_graph(input_spec, model_info)

    target_nodes = list(graph.nodes)
    num_nodes = len(target_nodes)
    logging.info(f"Node names: {target_nodes}")

    model_test_inputs = parse_evaluation_locations(input_spec)

    # print(model_info)
    # exit(1)
    #########################
    ## Run algorithms
    if input_spec['inference_type'] == "regression":
        logging.info("Performing Regression")
        #################
        ## Pytorch
        model = net.MFNetTorch(graph, roots)

        ## Training Setup
        loss_fns = net.construct_loss_funcs(model)

        data_loaders = model_info_to_dataloaders(model_info, graph.nodes)

        ## Train
        model.train(data_loaders, target_nodes, loss_fns)

        ## Evaluate and save to file
        for node in graph.nodes:
            test_pts = model_test_inputs[node]
            if test_pts is not None:
                logging.info(f"Evaluating model {node} at test points")

                x = torch.Tensor(test_pts.to_numpy())
                vals = model([x], [node])[0].detach().numpy()
                results = pd.DataFrame(vals, columns=model_info[node].train_out.columns)

                dirname = model_info[node].output_dir
                os.makedirs(dirname, exist_ok=True)
                filename = os.path.join(dirname, "evaluations.dat")
                results.to_csv(filename, sep=' ', index=False)

    elif run_type == "pyro":

        noise_var = args.noisevar[0]
        model = net_pyro.MFNetProbModel(graph, roots, noise_var=noise_var)

        alg = args.pyro_alg[0]

        ## Algorithm parameters
        num_samples = args.num_samples
        logging.info(f"Number of samples = {num_samples}")

        # MCMC
        num_chains = 1
        warmup = args.burnin

        # SVI
        adam_params = {"lr": 0.005, "betas": (0.95, 0.999)}
        num_steps = args.num_steps

        
        ## Data setup
        X = [d.x for d in datasets]
        Y = [d.y for d in datasets]
        
        if alg[:3] == 'svi':

            optimizer = Adam(adam_params)
            if alg == 'svi-normal':
                guide = AutoNormal(model)
            elif alg == 'svi-multinormal':
                guide = AutoMultivariateNormal(model)
            elif alg == 'svi-iafflow':
                hidden_dim = args.iaf_hidden_dim # list
                num_transforms = args.iaf_depth
                guide = AutoIAFNormal(model,
                                      hidden_dim=hidden_dim,
                                      num_transforms=num_transforms)
            else:
                logging.info(f"Algorithm \'{alg}\' is not recognized")
                exit(1)
                
            svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

            logging.info(f"Number of steps = {num_steps}")
            #do gradient steps
            for step in range(num_steps):
                elbo = svi.step(X, target_nodes, Y)
                if step % 100 == 0:
                    logging.info(f"Iteration {step}\t Elbo loss: {elbo}")

            predictive = Predictive(model, guide=guide, num_samples=num_samples)
            if eval_locs != None:
                # print(eval_locs.shape)
                pred = predictive([eval_locs]*num_nodes, target_nodes)
                # print(list(pred.keys()))
            else: # just predict on training points
                pred = predictive(X, target_nodes) 

            # param_samples = [print(k, v.squeeze().shape)
            #                  for k,v in pred.items() if k[:3] != "obs"]

            param_samples = {k: v.squeeze() for k,v in pred.items() if k[:3] != "obs"}
            vals = {k: v for k,v in pred.items() if k[:3] == "obs"}                
        

        elif alg == 'mcmc':

            nuts_kernel = NUTS(model, jit_compile=False, full_mass=True)
            mcmc = MCMC(
                nuts_kernel,
                num_samples=num_samples,
                warmup_steps=warmup,
                num_chains=num_chains,
            )
            print("\n")
            mcmc.run(X, target_nodes, Y)
            print("\n")

            param_samples = mcmc.get_samples()

            predictive = Predictive(model,  mcmc.get_samples())            
            if eval_locs != None:
                vals = predictive([eval_locs]*num_nodes, target_nodes)
            else: # predict on training points
                vals = predictive(X, target_nodes)

        else:
            logging.info(f"Algorithm \'{alg}\' is not recognized")
            exit(1)

        for ii, node in enumerate(target_nodes):
            v = vals[f"obs{node}"].T # x by num_samples
            fname = f"{save_evals_filename}_model{node}"
            logging.info(f"Saving outputs of Node {node} to: {fname}")
            np.savetxt(fname, v)



        # fig, axs = plt.subplots(3,1, sharex=True)
        # axs[0].plot(eval_locs, vals["obs1"].transpose(0,1), '-r', alpha=0.2)
        # axs[0].plot(X[0], Y[0], 'ko')

        # axs[1].plot(eval_locs, vals["obs2"].transpose(0,1), '-r', alpha=0.2)
        # axs[1].plot(X[1], Y[1], 'ko')

        # axs[2].plot(eval_locs, vals["obs3"].transpose(0,1), '-r', alpha=0.2)
        # axs[2].plot(X[2], Y[2], 'ko')

        # pred_filename = f"{save_evals_filename}_predict.pdf"
        # plt.savefig(pred_filename)
        # plt.show()
        
        df = net_pyro.samples_to_pandas(param_samples)

        logging.info(df.describe())

        sample_filename = f"{save_evals_filename}_param_samples.csv"
        logging.info(f"Saving samples to {sample_filename}")
        df.to_csv(sample_filename, index=False)

        # df = pd.read_csv(sample_filename)
        # print(df.describe())
            
