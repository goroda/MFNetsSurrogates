"""Command Line Utility for MFNETS."""
import sys
import yaml
import os

from collections import namedtuple

import argparse
import networkx as nx
import torch
import numpy as np
from sklearn import preprocessing
# import matplotlib.pyplot as plt

from mfnets_surrogates import net_torch as net
from mfnets_surrogates import net_pyro

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
        logging.info('Scale-shift edge functions are used')
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

    elif input_spec['graph']['connection_type'] == "general":
        logging.info('General edge functions are used')
        for node in graph.nodes:
            dim_in = model_info[node].dim_in
            dim_out = model_info[node].dim_out
            # print(list(graph.predecessors(node)
            num_inputs_parents = np.sum([model_info[p].dim_out for p in graph.predecessors(node)])

            logging.info(f'Assigning model for node {node}')
            # so far only use linear functions to test interface
            if num_inputs_parents == 0:
                for model in input_spec['graph']['connection_models']:
                    if model['name'] == node:
                        logging.info(f"Leaf node with type: {model['node_type']}")
                        if model['node_type'] == "linear":
                            graph.nodes[node]['func'] = torch.nn.Linear(dim_in, dim_out, bias=True)
                        elif model['node_type'] == "feedforward":
                            hidden_layer = model['hidden_layers']
                            logging.info(f'Feedforward with hidden layer sizes: {hidden_layer}')
                            graph.nodes[node]['func'] = net.FeedForwardNet(dim_in, dim_out,
                                                                           hidden_layer_sizes=hidden_layer)
                        else:
                            raise Exception(f"Node type {model.node_type} unknown")
                        break
                                
            else:
                for model in input_spec['graph']['connection_models']:
                    if model['name'] == node:
                        logging.info(f"Regular node with type: {model['node_type']}")
                        if model['edge_type'] == 'equal_model_average':
                            logging.info(f"Processing model averaged edge")
                            if model['node_type'] == "linear":
                                graph.nodes[node]['func'] = \
                                    net.EqualModelAverageEdge(dim_in, dim_out,
                                                              num_inputs_parents,
                                                              torch.nn.Linear(dim_in, dim_out, bias=True))
                            elif model['node_type'] == "feedforward":
                                hidden_layer = model['hidden_layers']
                                logging.info(f'Feedforward with hidden layer sizes: {hidden_layer}')
                                graph.nodes[node]['func'] = \
                                    net.EqualModelAverageEdge(dim_in, dim_out,
                                                              num_inputs_parents,
                                                              net.FeedForwardNet(dim_in, dim_out,
                                                                           hidden_layer_sizes=hidden_layer))
                            else:
                                raise Exception(f"Node type {model.node_type} unknown")
                        else:
                            logging.info(f"Processing learned edge")
                            if model['node_type'] == "linear":
                                graph.nodes[node]['func'] = net.LinearScaleShift(dim_in, dim_out, num_inputs_parents)
                            elif model['node_type'] == "feedforward":
                                hidden_layer = model['hidden_layers']
                                logging.info(f'Feedforward with hidden layer sizes: {hidden_layer}')
                                graph.nodes[node]['func'] = net.FullyConnectedNNEdge(dim_in, dim_out,
                                                                                     num_inputs_parents,
                                                                                     hidden_layer_sizes=hidden_layer)
                            else:
                                raise Exception(f"Node type {model.node_type} unknown")

                
            graph.nodes[node]['dim_in'] = dim_in
            graph.nodes[node]['dim_out'] = dim_out
            
            
    else:
        logging.error(f"Connection type {model_info['graph']['connection_type']} is not recognized")
        exit(1)

    return graph

def parse_graph(input_spec, model_info):
    """Parse the graph."""
    graph_file = input_spec['graph']['structure']

    try:
        with open(graph_file) as f:
            graph_read = f.read().splitlines()
    except FileNotFoundError:
        print(f"Cannot open file {graph_file}")
        exit(1)

    structure_format = "edge list"
    if "structure_format" in input_spec['graph']:
        structure_format = input_spec['graph']['structure_format']
    logging.info(f"Graph file type: {structure_format}")
    if structure_format == "edge list":
        graph = fill_graph(nx.parse_edgelist(graph_read, create_using=nx.DiGraph, nodetype=int), input_spec, model_info)
    elif structure_format == "adjacency list":
        graph = fill_graph(nx.parse_adjlist(graph_read, create_using=nx.DiGraph, nodetype=int), input_spec, model_info)
    else:
        logging.error(f"File type unrecognized")
        exit(1)

    # print(graph)
        
    roots = set([x for x in graph.nodes() if graph.in_degree(x) == 0])

    logging.info(f"Root models: {roots}")
    # exit(1)
    return graph, roots

def parse_model_info(args):
    """Parse data files."""
    num_models = input_spec['num_models']
    logging.info(f"Number of models: {num_models}")

    models = {}
    for model in input_spec['model_info']:

        name = model['name']

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

        output_dir = os.path.join(os.getcwd(), input_spec['save_dir'], model['output_dir'])
        
        models[name] = ModelTrainData(train_input, train_output, dim_in, dim_out, output_dir)
        logging.info(f"Model {name}: number of inputs = {dim_in}, number of outputs = {dim_out}, ntrain = {train_output.shape[0]}, output_dir = {output_dir}")

    return models

def parse_evaluation_locations(input_spec):
    """Parse eval_locations."""
    model_evals = {} 
    for ii, model in enumerate(input_spec['model_info']):

        name = model['name']

        if 'test_output' in model:

            filename = model['test_output']
            logging.info(f"Will evaluate model {name} at inputs of file(s) {filename}")

            if isinstance(filename, list):
                model_evals[name] = []
                for fname in filename:
                    try: 
                        test_input = pd.read_csv(fname, delim_whitespace=True)
                    except FileNotFoundError:
                        print(f"Cannot open test inputs for model {name} in file {fname}")
                        exit(1)
                    fname = fname.split(os.path.sep)[-1]
                    model_evals[name].append((fname, test_input))

            else:
                try: 
                    test_input = pd.read_csv(filename, delim_whitespace=True)
                except FileNotFoundError:
                    print(f"Cannot open test inputs for model {name} in file {filename}")
                    exit(1)
                    
                fname = filename.split(os.path.sep)[-1]
                model_evals[name] = [(fname, test_input)]
        else:
            model_evals[name] = None
            
    return model_evals

def model_info_to_dataloaders(model_info, graph_nodes):
    """Convert datasets to dataloaders for pytorch training."""
    data_loaders = []
    scalers_in = {}
    scalers_out ={}
    for node in graph.nodes:
        model = model_info[node]
        # print(model)
        x = model.train_in.to_numpy()
        if x.ndim == 1:
            x = x[:, np.newaxis]
        y = model.train_out.to_numpy()
        if y.ndim == 1:
            y = y[:, np.newaxis] 


        scaler_in = preprocessing.StandardScaler().fit(x)
        x_scaled = scaler_in.transform(x)

        scaler_out = preprocessing.StandardScaler().fit(y)
        y_scaled = scaler_out.transform(y)

        scalers_in[node]  = scaler_in
        scalers_out[node] = scaler_out
        # dataset = net.ArrayDataset(torch.Tensor(x), torch.Tensor(y))
        dataset = net.ArrayDataset(torch.Tensor(x_scaled), torch.Tensor(y_scaled))
        data_loaders.append(torch.utils.data.DataLoader(dataset, batch_size=x.shape[0], shuffle=False))        

    return data_loaders, scalers_in, scalers_out

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


    # print(input_spec)
    
    model_info = parse_model_info(input_spec)

    graph, roots = parse_graph(input_spec, model_info)

    target_nodes = list(graph.nodes)
    num_nodes = len(target_nodes)
    logging.info(f"Node names: {target_nodes}")

    model_test_inputs = parse_evaluation_locations(input_spec)

    save_dir = input_spec['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # print(model_info)
    # exit(1)
    #########################
    ## Run algorithms
    if input_spec['inference_type'] == "regression":
        logging.info("Performing Regression")
        #################
        ## Pytorch
        model = net.MFNetTorch(graph, roots, edge_type=input_spec['graph']['connection_type'])

        ## Training Setup
        loss_fns = net.construct_loss_funcs(model)

        data_loaders, scalers_in, scalers_out = model_info_to_dataloaders(model_info, graph.nodes)
        

        ## Train
        model.train(data_loaders, target_nodes, loss_fns)

        ## Evaluate and save to file
        for node in graph.nodes:
            test_pts = model_test_inputs[node]
            if test_pts is not None:
                logging.info(f"Evaluating model {node} at test points")

                for (fname, data) in test_pts:
                    # print("fname = ", fname)
                    x = torch.Tensor(data.to_numpy())
                    x_scaled = torch.Tensor(scalers_in[node].transform(x))
                    
                    vals = model([x_scaled], [node])[0].detach().numpy()
                    vals_unscaled = scalers_out[node].inverse_transform(vals)
                    results = pd.DataFrame(vals_unscaled, columns=model_info[node].train_out.columns)
                    dirname =  model_info[node].output_dir
                    os.makedirs(dirname, exist_ok=True)
                    filename = os.path.join(dirname, f"{fname}.evals")
                    results.to_csv(filename, sep=' ', index=False)

    elif input_spec['inference_type'] == "bayes": # UNTESTED

        logging.info("Running Bayesian Inference")
        noise_var = float(input_spec['algorithm']['noise_var'])

        model = net_pyro.MFNetProbModel(graph, roots, noise_var=noise_var,
                                        edge_type=input_spec['graph']['connection_type'],)

        alg = input_spec['algorithm']['parameterization']

        ## Algorithm parameters
        num_samples = input_spec['algorithm']['num_samples']
        logging.info(f"Number of parameter samples: {num_samples}")
        
        ## Data setup
        X = [torch.Tensor(model_info[node].train_in.to_numpy()) for node in target_nodes]
        Y = [torch.Tensor(model_info[node].train_out.to_numpy()) for node in target_nodes]
        if alg[:3] == 'svi':
            # SVI
            logging.info("Running Stochastic Variational Inference")
            
            adam_params = {"lr": 0.005, "betas": (0.95, 0.999)}
            num_steps = input_spec['algorithm']['num_optimization_steps']
        
            optimizer = Adam(adam_params)
            if alg == 'svi-normal':
                logging.info("Approximating with an AutoNormal Guide")
                guide = AutoNormal(model)
            elif alg == 'svi-multinormal':
                guide = AutoMultivariateNormal(model)
            elif alg == 'svi-iafflow':
                hidden_dim = input_spec['algorithm']['iaf_params']['hidden_dim']
                num_transforms = input_spec['algorithm']['iaf_params']['num_transforms']
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

            # just generate arbitrary input locations, we are just getting samples
            # of latent space here, this is a hack but unclear how to just sample
            # from the guide
            xone = [xx[0:1,:] for xx in X]
            pred = predictive(xone, target_nodes)

            param_samples = {k:v for k,v in pred.items() if k[:3] != "obs"}

        elif alg == 'mcmc':
            # MCMC
            num_chains = 1
            warmup = input_spec['algorithm']['mcmc_params']['burnin']
            
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

        else:
            logging.info(f"Algorithm \'{alg}\' is not recognized")
            exit(1)


        # Save samples
        # print(param_samples)
        df = net_pyro.samples_to_pandas(param_samples)
        # print(df.describe())
        sample_filename = os.path.join(save_dir, input_spec['algorithm']['sample_output_files'])
        logging.info(f"Saving samples to {sample_filename}")
        df.to_csv(sample_filename, sep=' ', index=False)


        # Perform predictions
        predictive = Predictive(model,  param_samples)

        ## Evaluate and save to file
        for node in graph.nodes:
            test_pts = model_test_inputs[node]
            if test_pts is not None:
                logging.info(f"Evaluating model {node} at test points")

                for (fname, data) in test_pts:
                    # print("fname = ", fname)
                    x = torch.Tensor(data.to_numpy())
                    vals = predictive([x], [node])
                    # print("values = ")
                    # print(vals)
                    for jj, qoi_name in enumerate(model_info[node].train_out.columns):
                        dirname = model_info[node].output_dir
                        filename = os.path.join(dirname, f"{fname}.samples_{qoi_name}")
                        logging.info(f"Saving samples for file {fname} and qoi {qoi_name} to {filename}")
                        key = f"obs{node}-{jj}"
                        col_vals = vals[key].detach().numpy() # num_samples x 
                        col_names = [f"x_{ii}" for ii in range(x.size(dim=0))]
                        results = pd.DataFrame(col_vals, columns=col_names)
                        # print(results.describe())

                        os.makedirs(dirname, exist_ok=True)
                        results.to_csv(filename, sep=' ', index=False)


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



        # df = pd.read_csv(sample_filename)
        # print(df.describe())
            
