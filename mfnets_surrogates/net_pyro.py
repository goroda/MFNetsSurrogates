"""Probabilistic MFNets."""

import functools
import itertools

import pandas as pd

import torch

import pandas as pd
from pyro.infer import Predictive
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO

import pyro
import pyro.distributions as dist
from pyro.nn.module import to_pyro_module_

# from mfnets_surrogates import net_torch as net
from mfnets_surrogates.net_torch import MFNetTorch, make_graph_2, make_graph_2gen, make_graph_2gen_nn, ArrayDataset

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
        convert_to_pyro(self.model)
        
    def forward(self, x, targets, y=None, zero_noise=False):
        """Evaluate model."""
        # print("x = ", x)
        means = self.model(x, targets)
        # print("god means = ", means)
        if y == None:
            # print("SHOULD NOT BE HERE YET")
            # exit(1)
            for ii, (m, xx) in enumerate(zip(means, x)):
                # print("m = ", m.size())
                # print("xx size = ", xx.size())                
                # with pyro.plate(f"data{ii+1}", xx.shape[0]):
                #     obs = pyro.sample(f"obs{ii+1}", dist.Normal(m.flatten(), self.sigma), obs=None)
                if zero_noise is False:
                    for jj in range(m.size(dim=1)):
                        obs = pyro.sample(f"obs{targets[ii]}-{jj}",
                                          dist.MultivariateNormal(
                                              m[:, jj],
                                              self.sigma * torch.eye(m.size(dim=0))), obs=None)
                else:
                    for jj in range(m.size(dim=1)):
                        obs = pyro.sample(f"obs{targets[ii]}-{jj}", dist.Delta(m[:, jj]))
        else:            
            for ii, (m, xx, yy) in enumerate(zip(means, x, y)):
                # print("m = ", m.size())
                # print("xx size = ", xx.size())
                # print("yy size = ", yy.size())
                for jj in range(yy.size(dim=1)):
                    
                    obs = pyro.sample(f"obs{ii+1}-{jj}",
                                      dist.MultivariateNormal(m[:, jj],
                                                              self.sigma * torch.eye(m.size(dim=0))), obs=yy[:, jj])
                    # with pyro.plate(f"data{ii+1}-{jj}", yy.size(dim=0)):
                    #     obs = pyro.sample(f"obs{ii+1}-{jj}", dist.Normal(m[:, jj], self.sigma), obs=yy[:, jj])
                    # for kk in range(yy.size(dim=1)):
                    # # print("m = ", m[jj, kk])
                    # # print("y = ", yy[jj, kk])
                    # # print("sigma = ", self.sigma)
                    #     obs = pyro.sample(f"obs{ii+1}-{jj}-{kk}",
                    #                       dist.Normal(m[jj, kk], self.sigma),
                    #                       obs=yy[jj, kk])
                        
                # with pyro.plate(f"data{ii+1}", xx.shape[0]):
                #     # obs = pyro.sample(f"obs{ii+1}", dist.Normal(m.flatten(), self.sigma), obs=yy.flatten())

        # print("return?")
        return [m for m in means]

    def train(self, data, targets, guide, num_steps=100, adam_lr=0.005, adam_betas=(0.95, 0.999)):
        """ Train the model. 

        Parameters
        ----------
        data: list of data loaders for the data associated with each node
        targets: list of target nodes (same length as data) 
        svi: pyro variational inference interface

        Returns
        -------
        guide: the updated guide

        """

        adam_params = {"lr": adam_lr, "betas": adam_betas}    
        optimizer = Adam(adam_params)
        svi = SVI(self, guide, optimizer, loss=Trace_ELBO())
        
        X = [d.dataset.x for d in data]
        Y = [d.dataset.y for d in data]
        print_frac = 0.1
        print_increment  = int(print_frac * num_steps)
        for step in range(num_steps):
            elbo = svi.step(X, targets, Y)
            if step % print_increment == 0:
                print(f"Iteration {step}\t Elbo loss: {elbo}")

        return guide

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
        
    

def run_single_output():
    
    torch.manual_seed(1)

    pyro.clear_param_store()
    graph, roots = make_graph_2()


    # model = MFNetProbModel(graph, roots, noise_var=1e-4)
    model = MFNetProbModel(graph, roots, noise_var=1e-2)    


    x = torch.linspace(-1,1,10).reshape(10, 1)
    targets = [1, 2]
    num_models = len(targets)


    # Plot Prior Predictive
    plt.figure()
    for ii in range(1000):
        evals = model([x]*num_models, targets)
        plt.plot(x, evals[0].flatten(), color='blue', alpha=0.2)
        plt.plot(x, evals[1].flatten(), color='red', alpha=0.2)
    # plt.show()
    # exit(1)

    # data
    data1 = x
    data2 = 2*x**2 + x
    data = [data1, data2]

    # algorithms
    # num_samples = 10000
    num_samples = 10000
    num_chains = 1
    # warmup_steps = 1000
    warmup_steps = 1000
    adam_params = {"lr": 0.005, "betas": (0.95, 0.999)}
    num_steps = 2000
    
    # testing
    xtest = torch.linspace(-1,1,100).reshape(100,1)


    run_svi = True
    run_mcmc = False
    
    if run_svi == True:
        optimizer = Adam(adam_params)
        # guide = AutoNormal(model)
        # guide = AutoMultivariateNormal(model)
        guide = AutoIAFNormal(model, hidden_dim=[100], num_transforms=4)
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

        #do gradient steps
        for step in range(num_steps):
            elbo = svi.step([x]*num_models, targets, data)
            if step % 100 == 0:
                print(f"Iteration {step}\t Elbo loss: {elbo}")

        predictive = Predictive(model, guide=guide, num_samples=num_samples)
        pred = predictive([xtest]*num_models, targets)
        param_samples = {k: v.reshape(num_samples) for k,v in pred.items() if k[:3] != "obs"}
        vals = {k: v for k,v in pred.items() if k[:3] == "obs"}
    # print("svi_samples = ", list(param_samples.keys()))
    # print(param_samples)
    # exit(1)

    if run_mcmc == True:

        nuts_kernel = NUTS(model, jit_compile=False, full_mass=True)
        mcmc = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            num_chains=num_chains,
        )
        mcmc.run([x]*num_models, targets, data)
        param_samples = mcmc.get_samples()        
        predictive = Predictive(model,  mcmc.get_samples())# , return_sites=("obs1", "_RETURN"))
        vals = predictive([xtest]*num_models, targets)
        
    
    # plt.figure()    
    # for ii in range(num_samples):
    #     # plt.plot(xtest.flatten(), predictive['_RETURN'][ii, :], '-r', alpha=0.2)
    #     plt.plot(xtest.flatten(), vals['obs1'][ii, :], 'blue', alpha=0.2)
    #     plt.plot(xtest.flatten(), vals['obs2'][ii, :], 'red', alpha=0.2)
    # plt.plot(x, data1, 'ko')
    # plt.plot(x, data2, 'mo')


    # df = samples_to_pandas(param_samples)
    # plt.figure()
    # plt.plot(df.iloc[:, 0])
    # plt.plot(df.iloc[:, 1])


    # pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(6,6), diagonal='kde')

    plt.show()


def run_multi_output_gen():
    
    torch.manual_seed(1)

    pyro.clear_param_store()

    # graph, roots = make_graph_2()
    # model = MFNetProbModel(graph, roots, noise_std=1e-4)

    # OR

    graph, roots = make_graph_2gen()    
    # graph, roots = make_graph_2gen_nn()
    model = MFNetProbModel(graph, roots, noise_std=1e-4, edge_type="general")    

    # exit(1)

    x = torch.linspace(-1,1,40).reshape(40, 1)
    targets = [1, 2]
    num_models = len(targets)


    # Plot Prior Predictive
    plt.figure()
    for ii in range(1000):
        evals = model([x]*num_models, targets)
        plt.plot(x, evals[0].flatten(), color='blue', alpha=0.2)
        plt.plot(x, evals[1].flatten(), color='red', alpha=0.2)
    plt.title("Prior predictive")
    # plt.show()
    # exit(1)

    # data
    data1 = x
    data2 = 2*x**2 + x
    data = (ArrayDataset(x.clone().detach(), data1), ArrayDataset(x.clone().detach(), data2))
    data_loaders = [torch.utils.data.DataLoader(d, batch_size=len(d), shuffle=False)
                    for d in data]
    
    # testing
    xtest = torch.linspace(-1,1,100).reshape(100,1)


    guide = AutoDelta(model) # DOESNT WORK
    # guide = AutoNormal(model)
    # guide = AutoMultivariateNormal(model)
    # guide = AutoIAFNormal(model, hidden_dim=[20, 20, 20, 20, 20], num_transforms=10)
    # guide = model.train(data_loaders, targets, guide, adam_lr=0.1, num_steps=200) # good for scale-shift
    guide = model.train(data_loaders, targets, guide, adam_lr=0.01, num_steps=1000) # 1000 may be good for nn

    num_samples = 1000
    df_params, df_obs = model.sample_pred([xtest]*num_models, targets, guide, num_samples)

    

    print(df_params)
    print(df_obs)

    plt.figure()
    model1_output = model.extract_model_output_from_samples(df_obs, 1, 0)
    model2_output = model.extract_model_output_from_samples(df_obs, 2, 0)
    plt.plot(xtest.flatten(), model1_output.T, 'red', alpha=0.2)
    plt.plot(xtest.flatten(), xtest.flatten(), 'k-')

    plt.plot(xtest.flatten(), model2_output.T, 'blue', alpha=0.2)
    plt.plot(xtest.flatten(), 2*xtest.flatten()**2 + xtest.flatten(), 'k-')    
    
    pd.plotting.scatter_matrix(df_params, alpha=0.2, figsize=(6,6), diagonal='kde')

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


    # run_single_output()
    run_multi_output_gen()
