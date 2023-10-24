"""Probabilistic MFNets."""

import functools
import itertools

import pandas as pd

import torch
import numpy as np

import pandas as pd
from pyro.infer import Predictive
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO

import pyro
import pyro.distributions as dist
from pyro.nn.module import to_pyro_module_

# from mfnets_surrogates import net_torch as net
from mfnets_surrogates.net_torch import MFNetTorch, make_graph_2, make_graph_2gen, make_graph_2gen_nn, make_graph_2gen_nn_fixed, make_graph_single, ArrayDataset, generate_data, generate_data_1model

# from pandas.tools.plotting import scatter_matrix
# import pandas.tools.plotting as pandaplot
# try: 
#     from .net_torch import MFNetTorch, make_graph_2, make_graph_2gen, make_graph_2gen_nn, ArrayDataset
# except:


__all__ = [
    "MFNetProbModel",
    "samples_to_pandas"
]
    

def nestgetattr(obj, name):
    """Nested getattr."""
    def func(current, new):
        return getattr(current, new)
    
    res = functools.reduce(func, name.split("."), obj)
    return res

def nestsetattr(obj, name, value):
    """Nested setattr."""
    def func(current, new):
        return getattr(current, new)

    names = name.split(".")
    if len(names) == 1:
        setattr(obj, name, value)
    else:
        last_obj = functools.reduce(func, names[:-1], obj)
        setattr(last_obj, names[-1], value)
    
def convert_to_pyro(model):
    """Convert an MFNetTorch model to a Pyro Model.

    Also sets all internal parameters to uncertain random variables (priors)
    """
    # print(model)
    to_pyro_module_(model)
    # print(model)
    names = []
    for name, param in model.named_parameters():
        # print(name, param)
        names.append(name)


    for name in names:
        # print("NAME = ", name)
        param = nestgetattr(model, name)
        size = list(param.size())
        dim = param.dim()
        # print(size, dim)
        new_param = pyro.nn.PyroSample(
            dist.Normal(0,1).expand(size).to_event(dim))
        # print(f"\t size={size}, dim={dim}")
        nestsetattr(model, name, new_param)


    # print("\n\n\n")        
    # for name, param in model.named_parameters():
    #     print(name, param)
            # print(name, )
    # print("params after:", [name for name, _ in model.named_parameters()])

def param_samples_to_pandas(samples):
    hmc_samples = {k: v.detach().cpu().numpy() for k, v in samples.items()}
    names = list(hmc_samples.keys())
    param_dict = {}
    for key,val in samples.items():
        val = val.squeeze()
        
        if val.dim() == 1:
            val = val.reshape(val.size(dim=0), 1)

        name = key.split("modules_list.")
        name = name[-1]
        # v = val.detach().numpy()
        # assert v.ndim == 2
        for idx in itertools.product(*map(range, val.shape[1:])): # first dimension is samples

            idx_str = "[{}]".format(",".join(map(str, idx)))
            new_name = name + idx_str
            if len(idx) == 1:
                v = val[:, idx[0]]
            elif len(idx) == 2:
                v = val[:, idx[0], idx[1]]
            else:
                print("not sure  dimension is larger than 2")
                # print("val shape = ", val.shape)
                exit(1)
                # print("vv = ", v.shape)
            param_dict[new_name] = v.detach().numpy()            
    
    df_params = pd.DataFrame(param_dict)
    return df_params

def samples_to_pandas(samples):
    """Convert the samples output of Pyro parameters to a pandas dataframe."""
    hmc_samples = {k: v.detach().cpu().numpy() for k, v in samples.items()}
    names = list(hmc_samples.keys())
    param_dict = {}
    # print(names)

    obs_dict = {}
    for key,val in samples.items():
        val = val.squeeze()
        
        if val.dim() == 1:
            val = val.reshape(val.size(dim=0), 1)

        # print(key)
        # print("\nval shape = ", val.shape)
            

        if 'obs' in key:
            name = key.split('obs')
            model_out = name[1].split("-")
            new_name = f"model_{model_out[0]}_output_{model_out[1]}"
            # print("val.shape = ", val.shape)
            v = val.detach().numpy()
            assert val.ndim == 2
            for jj in range(v.shape[1]):
                out_name = f"{new_name}_pred_at_input{jj}"
                obs_dict[out_name] = v[:, jj]
            
        else:
            name = key.split("modules_list.")
            name = name[-1]
            # v = val.detach().numpy()
            # assert v.ndim == 2
            for idx in itertools.product(*map(range, val.shape[1:])): # first dimension is samples

                idx_str = "[{}]".format(",".join(map(str, idx)))
                new_name = name + idx_str
                if len(idx) == 1:
                    v = val[:, idx[0]]
                elif len(idx) == 2:
                    v = val[:, idx[0], idx[1]]
                else:
                    print("not sure  dimension is larger than 2")
                    # print("val shape = ", val.shape)
                    exit(1)
                # print("vv = ", v.shape)
                param_dict[new_name] = v.detach().numpy()

    # print(param_dict)
    # print(list(param_dict.keys()))

    df_params = pd.DataFrame(param_dict)
    # print(df_params.columns)
    # exit(1)
    df_obs = pd.DataFrame(obs_dict)
    # print(df)
    # print(df.describe())
    return df_params, df_obs
    

class MFNetProbModel(pyro.nn.PyroModule):
    """Probabilistic MFNet."""
    
    def __init__(self, graph, roots, noise_std=1.0, **kwargs):
        """Initialize Probabilistic MFNET."""
        super().__init__()
        self.model = MFNetTorch(graph, roots, **kwargs)
        # print(self.model)
        self.sigma = noise_std
        self.guide = None # for variational inference
        self.mcmc = None # for mcmc
        convert_to_pyro(self.model)
        
    def forward(self, x, targets, y=None):
        """Evaluate model."""

        means = self.model(x, targets)
        # print("god means = ", means)
        if y == None:
            # print("SHOULD NOT BE HERE YET")
            # exit(1)
            for ii, (m, xx) in enumerate(zip(means, x)):
                dout = m.shape[1]
                cov = self.sigma**2 * torch.eye(dout)
                for jj in range(m.shape[0]):
                    obs = pyro.sample(f"obs/{targets[ii]}/{jj}",
                                      dist.MultivariateNormal(m[jj, :], cov),
                                      obs=None)
                # else:
                #     for jj in range(m.shape[0]):
                #         obs = pyro.sample(f"obs/{targets[ii]}/{jj}",
                #                           dist.Delta(m[jj, :]))
        else:
            # print("y = ", y)
            for ii, (m, xx, yy) in enumerate(zip(means, x, y)):

                # print("xx size = ", xx.size())
                # print("yy size = ", yy.size())
                # print("num_data = ", yy.shape[0])                
                dout = yy.shape[1]
                cov = self.sigma**2 * torch.eye(dout)
                for jj in range(yy.shape[0]):
                    obs = pyro.sample(
                        f"obs/{ii+1}/{jj}",
                        dist.MultivariateNormal(m[jj, :], cov),
                        obs=yy[jj, :])
        return [m for m in means]

    def train_svi(self,
                  data,
                  targets,
                  guide,
                  adam_params={"lr": 0.01,
                               "betas": (0.95, 0.999)},
                  max_steps=1000,
                  print_frac=0.1):
        """ Train the model. 

        Parameters
        ----------
        data: list of data loaders for the data associated with each node
        targets: list of target nodes (same length as data)
        guide: guide to use (this is then stored internally)
        adam_params: parameters to use for optimization
        max_steps: number of optimization steps to take
        print_frac: printing frequency

        Returns
        -------
        self
        """

        optimizer = Adam(adam_params)
        svi = SVI(self, guide, optimizer, loss=Trace_ELBO())
        self.guide = guide
        x = []
        y = []
        for dat in data:
            assert len(dat) == 1
            for batch, (X, Y) in enumerate(dat):
                x.append(X)
                y.append(Y)        
        
        print_increment  = int(print_frac * max_steps)
        for step in range(max_steps):
            elbo = svi.step(x, targets, y)
            if step % print_increment == 0:
                print(f"Iteration {step}\t Elbo loss: {elbo}")
                
        return self

    def train_mcmc(self,
                   data,
                   targets,
                   num_samples=1000,
                   warmup_frac=0.1,
                   num_chains=1):
        """ Train the model. 

        Parameters
        ----------
        data: list of data loaders for the data associated with each node
        targets: list of target nodes (same length as data)
        guide: guide to use (this is then stored internally)
        adam_params: parameters to use for optimization
        max_steps: number of optimization steps to take
        print_frac: printing frequency

        Returns
        -------
        self

        Note
        ------
        Stores n mcmc object
        """

        assert num_chains == 1, "Havent implemented merging chains yet in predict, so dont generate data with multiple chains here"
        x = []
        y = []
        for dat in data:
            assert len(dat) == 1
            for batch, (X, Y) in enumerate(dat):
                x.append(X)
                y.append(Y)

        warmup_steps = int(warmup_frac * num_samples)
        nuts_kernel = NUTS(self, jit_compile=False, full_mass=True)
        mcmc = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            num_chains=num_chains,
        )
        mcmc.run(x, targets, y)
        self.mcmc = mcmc
        
        return self    
           
    def sample_pred(self, xpred, targets, guide, num_samples):
        """Make predictions by sampling from the guide and running it through the model.

        Parameters
        ----------
        xpred: locations at which to predict
        targets: list of target nodes (same length as xpred) 
        guide: the guide used to generate samples
        num_samples: the number of samples to generate
        
        Returns
        -------
        df_params: Pandas DataFrame of parameter samples
        df_obs: Pandas DataFrame of output samples
        """

        predictive = Predictive(self, guide=guide, num_samples=num_samples)
        pred = predictive(xpred, targets, zero_noise=True)
        df_params, df_obs = samples_to_pandas(pred)
        return df_params, df_obs
        
    def extract_model_output_from_samples(self, df_samples, model_num, output_num):
        output_cols = df_samples.columns


        desired_cols = [d for d in output_cols if f'model_{model_num}_output_{output_num}' in d]
        df_out = df_samples[desired_cols]

        return df_out.to_numpy()

    def predict(self, xpred, targets, num_samples):
        """Generate predictions.
        Parameters
        ---------
        xpred: list of locations to predict for models for which predictions are desired
        targets: list of models to predict
        guide: guide to use for predictions
        num_samples: number of samples to predict

        Returns
        -------
        param_samples: samples of the number of parmaeters
        pred_samples: list of size of targets corresponding to predictions
                     pred_samples = num_samples x num_outputs (NEED TO CHECK FOR MULTI-OUTPUT)
       
        """
        if self.guide != None:
            predictive = Predictive(self, guide=self.guide, num_samples=num_samples)
        elif self.mcmc != None:
            predictive = Predictive(self,  self.mcmc.get_samples())
   
        else:
            raise "must run either SVI or MCMC"

        pred = predictive(xpred, targets)

        if self.guide != None:
            param_samples = {k: v for k,v in pred.items() if k[:3] != "obs"}
        elif self.mcmc != None:
            param_samples = self.mcmc.get_samples()
            
        df_params = param_samples_to_pandas(param_samples)        
        pred_samples = [None] * len(targets)
        for ii in range(len(targets)):
            pred_samples[ii] = [v for k,v in pred.items() if k.startswith(f"obs/{targets[ii]}/")]

            pred_samples[ii] = torch.cat(pred_samples[ii], dim=-1) # might change for multi output

        return df_params, pred_samples
        
        
def run_scale_shift():

    torch.manual_seed(1)
    pyro.clear_param_store()


    # Create the data
    graph, root = make_graph_2()
    model = MFNetTorch(graph, root)

    data = generate_data(model, [20, 20])
    data_loaders = [torch.utils.data.DataLoader(d, batch_size=len(d), shuffle=False)
                    for d in data]


    # Now train
    num_models = 2
    graph, roots = make_graph_2()
    model_trained = MFNetProbModel(graph, roots, noise_std=1e-5)

    # Plot Prior Predictive
    plt.figure()
    targets = [1, 2]
    x = torch.linspace(-1,1,10).reshape(10, 1)
    for ii in range(1000):        
        evals = model_trained([x]*num_models, targets)
        plt.plot(x, evals[0].flatten(), color='blue', alpha=0.2)
        plt.plot(x, evals[1].flatten(), color='red', alpha=0.2)
    

    # TRAIN
    num_samples = 1000
    run_svi = False
    run_mcmc = True
    
    if run_svi == True:
        num_steps = 1000
        adam_params = {"lr": 0.005, "betas": (0.95, 0.999)}
        
        guide = AutoNormal(model_trained)
        # guide = AutoMultivariateNormal(model_trained)
        # guide = AutoIAFNormal(model_trained, hidden_dim=[100], num_transforms=4)
        model_trained.train_svi(data_loaders,
                                targets, guide, adam_params, max_steps=num_steps)
        
    if run_mcmc == True:
        num_chains = 1
        warmup_frac = 0.2
        model_trained.train_mcmc(data_loaders,
                                 targets, num_samples=num_samples,
                                 warmup_frac=warmup_frac)

        
    # TEST
    xtest = torch.linspace(-3,3,100).reshape(100,1)
    
    param_samples, pred_samples = model_trained.predict([xtest]*num_models,
                                                        targets, num_samples)
    #pred_samples = num_samples x num_outputs    
    plt.figure()
    for ii in range(num_samples):
        plt.plot(xtest.flatten(), pred_samples[0][ii, :], 'blue', alpha=0.2)
        plt.plot(xtest.flatten(), pred_samples[1][ii, :], 'red', alpha=0.2)
        
    plt.plot(data[0].x, data[0].y, 'ko')
    plt.plot(data[1].x, data[1].y, 'mo')


    print(param_samples)
    # df = samples_to_pandas(param_samples)
    plt.figure()
    plt.plot(param_samples.iloc[:, 0])
    plt.plot(param_samples.iloc[:, 1])


    pd.plotting.scatter_matrix(param_samples, alpha=0.2, figsize=(6,6), diagonal='kde')

    plt.show()


def run_generalized():
    
    torch.manual_seed(1)

    pyro.clear_param_store()

    # Generate Data
    graph, root = make_graph_2gen()
    model = MFNetTorch(graph, root, edge_type="general")
    x = torch.linspace(-3, 3, 10).reshape(10, 1)
    # plot_funcs(model, x, title="True Model")

    data = generate_data(model, [60, 60])
    data_loaders = [torch.utils.data.DataLoader(d, batch_size=len(d), shuffle=False)
                    for d in data]
    
    # graph, roots = make_graph_2()
    # model = MFNetProbModel(graph, roots, noise_std=1e-4)

    # Train
    graph2, root2 = make_graph_2gen()
    model_trained = MFNetProbModel(graph2, root2, noise_std=1e-4, edge_type="general")
    targets = [1, 2]
    num_models = len(targets)


    # Plot Prior Predictive
    plt.figure()
    x = torch.linspace(-1,1,40).reshape(40, 1)
    for ii in range(1000):
        evals = model_trained([x]*num_models, targets)
        plt.plot(x, evals[0].detach().numpy().flatten(), color='blue', alpha=0.2)
        plt.plot(x, evals[1].detach().numpy().flatten(), color='red', alpha=0.2)
    plt.title("Prior predictive")


    # Train
    # guide = AutoNormal(model_trained)
    guide = AutoIAFNormal(model_trained, hidden_dim=[20], num_transforms=2)    
    model_trained.train_svi(data_loaders, targets, guide, max_steps=1000)
    
    # Test
    num_samples = 1000
    xtest = torch.linspace(-3,3,100).reshape(100,1)
    param_samples, pred_samples = model_trained.predict([xtest]*num_models,
                                                        targets, num_samples)
    #pred_samples = num_samples x num_outputs    
    plt.figure()
    for ii in range(num_samples):
        plt.plot(xtest.flatten(), pred_samples[0][ii, :], 'blue', alpha=0.2)
        plt.plot(xtest.flatten(), pred_samples[1][ii, :], 'red', alpha=0.2)
        
    plt.plot(data[0].x, data[0].y, 'ko')
    plt.plot(data[1].x, data[1].y, 'mo')


    print(param_samples)
    # df = samples_to_pandas(param_samples)
    plt.figure()
    plt.plot(param_samples.iloc[:, 0])
    plt.plot(param_samples.iloc[:, 1])


    pd.plotting.scatter_matrix(param_samples, alpha=0.2, figsize=(6,6), diagonal='kde')

    plt.show()

def run_generalized_nn():
    
    torch.manual_seed(1)

    pyro.clear_param_store()

    # Generate Data
    graph, root = make_graph_2gen()
    model = MFNetTorch(graph, root, edge_type="general")

    # plot_funcs(model, x, title="True Model")

    data = generate_data(model, [20, 20])
    data_loaders = [torch.utils.data.DataLoader(d, batch_size=len(d), shuffle=False)
                    for d in data]
    
    # graph, roots = make_graph_2()
    # model = MFNetProbModel(graph, roots, noise_std=1e-4)

    # Train
    graph2, root2 = make_graph_2gen_nn()
    model_trained = MFNetProbModel(graph2, root2, noise_std=1e-4, edge_type="general")
    targets = [1, 2]
    num_models = len(targets)


    # Plot Prior Predictive
    plt.figure()
    x = torch.linspace(-1,1,40).reshape(40, 1)
    # x = torch.linspace(-3, 3, 10).reshape(10, 1)    
    for ii in range(1000):
        evals = model_trained([x]*num_models, targets)
        plt.plot(x, evals[0].detach().numpy().flatten(), color='blue', alpha=0.2)
        plt.plot(x, evals[1].detach().numpy().flatten(), color='red', alpha=0.2)
    plt.title("Prior predictive")


    # Train
    guide = AutoNormal(model_trained)
    # guide = AutoIAFNormal(model_trained, hidden_dim=[20], num_transforms=2)    
    model_trained.train_svi(data_loaders, targets, guide, max_steps=10000)
    
    # Test
    num_samples = 1000
    xtest = torch.linspace(-1,1,100).reshape(100,1)
    param_samples, pred_samples = model_trained.predict([xtest]*num_models,
                                                        targets, num_samples)
    #pred_samples = num_samples x num_outputs    
    plt.figure()
    for ii in range(num_samples):
        plt.plot(xtest.flatten(), pred_samples[0][ii, :], 'blue', alpha=0.2)
        plt.plot(xtest.flatten(), pred_samples[1][ii, :], 'red', alpha=0.2)
        
    plt.plot(data[0].x, data[0].y, 'ko')
    plt.plot(data[1].x, data[1].y, 'mo')


    # print(param_samples.summary())
    # df = samples_to_pandas(param_samples)
    # plt.figure()
    # plt.plot(param_samples.iloc[:, 0])
    # plt.plot(param_samples.iloc[:, 1])


    # pd.plotting.scatter_matrix(param_samples, alpha=0.2, figsize=(6,6), diagonal='kde')

    plt.show()

def run_single_fidelity():
    
    torch.manual_seed(1)

    pyro.clear_param_store()

    # Generate Data
    graph, root = make_graph_single() # train with non nn


    model = MFNetTorch(graph, root, edge_type="general")
    # plot_funcs(model, x, title="True Model")
    data = generate_data_1model(model, 20)
    data_loaders = [torch.utils.data.DataLoader(d, batch_size=len(d), shuffle=False)
                    for d in data]
    
    # graph, roots = make_graph_2()
    # model = MFNetProbModel(graph, roots, noise_std=1e-4)

    # Train
    graph2, root2 = make_graph_single()
    model_trained = MFNetProbModel(graph2, root2, noise_std=1e-4, edge_type="general")
    targets = [1]
    num_models = len(targets)


    # Plot Prior Predictive
    plt.figure()
    x = torch.linspace(-3,3,40).reshape(40, 1)
    for ii in range(1000):
        evals = model_trained([x]*num_models, targets)
        plt.plot(x, evals[0].detach().numpy().flatten(), color='blue', alpha=0.2)
    plt.title("Prior predictive")


    # Train
    guide = AutoNormal(model_trained)
    # guide = AutoIAFNormal(model_trained, hidden_dim=[20], num_transforms=2)    
    model_trained.train_svi(data_loaders, targets, guide, max_steps=14000)
    
    # Test
    num_samples = 1000
    xtest = torch.linspace(-3,3,100).reshape(100,1)
    param_samples, pred_samples = model_trained.predict([xtest]*num_models,
                                                        targets, num_samples)
    #pred_samples = num_samples x num_outputs    
    plt.figure()
    for ii in range(num_samples):
        plt.plot(xtest.flatten(), pred_samples[0][ii, :], 'blue', alpha=0.2)
        
    plt.plot(data[0].x, data[0].y, 'ko')


    # print(param_samples.summary())
    # df = samples_to_pandas(param_samples)
    # plt.figure()
    # plt.plot(param_samples.iloc[:, 0])
    # plt.plot(param_samples.iloc[:, 1])


    # pd.plotting.scatter_matrix(param_samples, alpha=0.2, figsize=(6,6), diagonal='kde')

    plt.show()        
    

def run_generalized_nn_fixed_edge():

    torch.manual_seed(1)
    graph, root = make_graph_2gen_nn_fixed() # train with non nn
    model = MFNetTorch(graph, root, edge_type="general")

    data = generate_data(model, [20, 20])
    data_loaders = [torch.utils.data.DataLoader(d, batch_size=len(d), shuffle=False)
                    for d in data]
    
    # graph, roots = make_graph_2()
    # model = MFNetProbModel(graph, roots, noise_std=1e-4)

    # Train
    graph2, root2 = make_graph_2gen_nn_fixed()
    model_trained = MFNetProbModel(graph2, root2, noise_std=1e-4, edge_type="general")
    targets = [1, 2]
    num_models = len(targets)


    # Plot Prior Predictive
    plt.figure()
    x = torch.linspace(-3,3,40).reshape(40, 1)
    for ii in range(1000):
        evals = model_trained([x]*num_models, targets)
        plt.plot(x, evals[0].detach().numpy().flatten(), color='blue', alpha=0.2)
        plt.plot(x, evals[1].detach().numpy().flatten(), color='red', alpha=0.2)
    plt.title("Prior predictive")


    # Train
    guide = AutoNormal(model_trained)
    # guide = AutoIAFNormal(model_trained, hidden_dim=[20], num_transforms=2)    
    model_trained.train_svi(data_loaders, targets, guide, max_steps=10000)
    
    # Test
    num_samples = 1000
    xtest = torch.linspace(-3,3,100).reshape(100,1)
    param_samples, pred_samples = model_trained.predict([xtest]*num_models,
                                                        targets, num_samples)
    #pred_samples = num_samples x num_outputs    
    plt.figure()
    for ii in range(num_samples):
        plt.plot(xtest.flatten(), pred_samples[0][ii, :], 'blue', alpha=0.2)
        plt.plot(xtest.flatten(), pred_samples[1][ii, :], 'red', alpha=0.2)
        
    plt.plot(data[0].x, data[0].y, 'ko')
    plt.plot(data[1].x, data[1].y, 'mo')


    # print(param_samples.summary())
    # df = samples_to_pandas(param_samples)
    # plt.figure()
    # plt.plot(param_samples.iloc[:, 0])
    # plt.plot(param_samples.iloc[:, 1])


    # pd.plotting.scatter_matrix(param_samples, alpha=0.2, figsize=(6,6), diagonal='kde')

    plt.show()

           
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import seaborn as sns


    from pyro.infer import MCMC, NUTS

    from pyro.infer.autoguide import (
        AutoDelta,
        AutoNormal,
        AutoMultivariateNormal,
        AutoLowRankMultivariateNormal,
        AutoGuideList,
        AutoIAFNormal,
        init_to_feasible,
    )


    # run_scale_shift()
    # run_generalized()
    # run_generalized_nn()
    # run_multi_output_gen()
    # run_single_fidelity()
    run_generalized_nn_fixed_edge()
