"""Probabilistic MFNets."""

import functools
import itertools
import torch
import pyro
import net_torch as net
import pyro.distributions as dist
from pyro.nn.module import to_pyro_module_
from pyro.infer import MCMC, NUTS
from pyro.infer import Predictive
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# from pandas.tools.plotting import scatter_matrix
# import pandas.tools.plotting as pandaplot


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
    to_pyro_module_(model)
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

class MFNetProbModel(pyro.nn.PyroModule):
    """Probabilistic MFNet."""
    
    def __init__(self, graph, roots, noise_var=1.0):
        """Initialize Probabilistic MFNET."""
        super().__init__()
        self.model = net.MFNetTorch(graph, roots)
        self.sigma = noise_var
        convert_to_pyro(self.model)
        
    def forward(self, x, targets, y=None):
        """Evaluate model."""
        means = self.model(x, targets)
        if y == None:
            for ii, (m, xx) in enumerate(zip(means, x)):
                with pyro.plate(f"data{ii+1}", xx.shape[0]):
                    obs = pyro.sample(f"obs{ii+1}", dist.Normal(m.flatten(), self.sigma), obs=None)
        else:
            for ii, (m, xx, yy) in enumerate(zip(means, x, y)):
                with pyro.plate(f"data{ii+1}", xx.shape[0]):
                    obs = pyro.sample(f"obs{ii+1}", dist.Normal(m.flatten(), self.sigma), obs=yy)
        return [m.flatten() for m in means]

def mcmc_samples_to_pandas(samples):
    """Convert the samples of parameters to a pandas dataframe."""
    
    hmc_samples = {k: v.detach().cpu().numpy() for k, v in samples.items()}
    names = list(hmc_samples.keys())
    new_dict = {}
    # print(names)
    for key,val in samples.items():

        # print(key)
        # print("\nval shape = ", val.shape)
        if 'node' in key:
            name = key.split('node')
            for idx in itertools.product(*map(range, val.shape[1:])): # first dimension is samples
                idx_str = "[{}]".format(",".join(map(str, idx)))
                new_name = 'node' + name[-1] + idx_str
                # print("name = ", new_name)
                # print("idx = ", idx)
                if len(idx) == 1:
                    v = val[:, idx[0]]
                elif len(idx) == 2:
                    v = val[:, idx[0], idx[1]]
                else:
                    print("not sure why dimension is larger than 2")
                    exit(1)
                # print("vv = ", v.shape)
                new_dict[new_name] = v.detach().numpy()

        if 'edge' in key:
            name = key.split('edge')
            for idx in itertools.product(*map(range, val.shape[1:])): # first dimension is samples
                idx_str = "[{}]".format(",".join(map(str, idx)))
                new_name = 'edge' + name[-1] + idx_str
                # print("name = ", new_name)
                # print("idx = ", idx)
                if len(idx) == 1:
                    v = val[:, idx[0]]
                elif len(idx) == 2:
                    v = val[:, idx[0], idx[1]]
                else:
                    print("not sure why dimension is larger than 2")
                    exit(1)
                # print("vv = ", v.shape)
                new_dict[new_name] = v.detach().numpy()

    # print(new_dict)
    # print(list(new_dict.keys()))

    df = pd.DataFrame(new_dict)
    # print(df)
    # print(df.describe())
    return df
        
if __name__ == "__main__":

    torch.manual_seed(1)

    pyro.clear_param_store()
    graph, roots = net.make_graph_2()


    model = MFNetProbModel(graph, roots, noise_var=1e-4)


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
    
    num_samples = 1000
    num_chains = 1
    warmup_steps = 50
    nuts_kernel = NUTS(model, jit_compile=False)
    mcmc = MCMC(
        nuts_kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
    )

    data1 = x
    data2 = 2*x**2 + x
    data = [data1.flatten(), data2.flatten()]
    mcmc.run([x]*num_models, targets, data)
    # mcmc.summary(prob=0.5)

    xtest = torch.linspace(-1,1,100).reshape(100,1)
    predictive = Predictive(model,  mcmc.get_samples())# , return_sites=("obs1", "_RETURN"))
    vals = predictive([xtest]*num_models, targets)

    plt.figure()    
    for ii in range(num_samples):
        # plt.plot(xtest.flatten(), predictive['_RETURN'][ii, :], '-r', alpha=0.2)
        plt.plot(xtest.flatten(), vals['obs1'][ii, :], 'blue', alpha=0.2)
        plt.plot(xtest.flatten(), vals['obs2'][ii, :], 'red', alpha=0.2)
    plt.plot(x, data1, 'ko')
    plt.plot(x, data2, 'mo')


    df = mcmc_samples_to_pandas(mcmc.get_samples())
    plt.figure()
    plt.plot(df.iloc[:, 0])
    plt.plot(df.iloc[:, 1])
    

    
    pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(6,6), diagonal='kde')

    plt.show()    
