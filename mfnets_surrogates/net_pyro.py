"""Probabilistic MFNets."""

import functools

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
            for ii, (m, xx) in enumerate(zip(means, xx)):
                with pyro.plate(f"data{ii+1}", xx.shape[0]):
                    obs = pyro.sample(f"obs{ii+1}", dist.Normal(m.flatten(), self.sigma), obs=None)
        else:
            for ii, (m, xx, yy) in enumerate(zip(means, xx, y)):
                with pyro.plate(f"data{ii+1}", xx.shape[0]):
                    obs = pyro.sample(f"obs{ii+1}", dist.Normal(m.flatten(), self.sigma), obs=yy)
        return means

if __name__ == "__main__":

    torch.manual_seed(1)

    pyro.clear_param_store()
    graph, roots = net.make_graph_2()

    # model = MFNetPyro(graph, root)
    # model = net.MFNetTorch(graph, root)    
    # model.set_target_node(2)
    model = MFNetProbModel(graph, roots, noise_var=1e-4)
    # model.model.set_target_node(2)

    x = torch.linspace(-1,1,10).reshape(10, 1)
    # with pyro.poutine.trace() as tr:
    #     model(example_input)

    # for site in tr.trace.nodes.values():
    #     print(site["type"], site["name"], site["value"])

    plt.figure()
    for ii in range(1000):
        plt.plot(x, model(x), color='blue', alpha=0.2)


    num_samples = 500
    num_chains = 1
    warmup_steps = 100
    nuts_kernel = NUTS(model, jit_compile=False)
    mcmc = MCMC(
        nuts_kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
    )

    data = 2*x**2 + x
    mcmc.run(x, data.flatten())
    mcmc.summary(prob=0.5)
    
    xtest = torch.linspace(-1,1,100).reshape(100,1)
    predictive = Predictive(model,  mcmc.get_samples(),
                            return_sites=("obs", "_RETURN"))(xtest, None)
    # print(predictive)
    # for k, v in predictive.items():
    #     print(f"{k}: {tuple(v.shape)}")
    # print(predictive['obs'])
    pred_vals = predictive["_RETURN"]
    print(pred_vals.size())
    plt.figure()    
    for ii in range(num_samples):
        plt.plot(xtest.flatten(), predictive['_RETURN'][ii, :], '-r', alpha=0.2)
    plt.plot(x, data, 'ko')

    
    hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
    names = list(hmc_samples.keys())
    print(names)
    # print(hmc_samples)
    plt.figure()
    plt.plot(hmc_samples[names[0]].flatten(), hmc_samples[names[1]].flatten(), 'o')


    plt.show()    
    # df = pd.DataFrame(hmc_samples)
    
    # print(df)
    # plt.show()




    
    # print(model(example_input))
    # print(model(example_input))
    # with pyro.poutine.trace() as tr:
    #     print(model(example_input))
    # print(type(model).__name__)
    # print(list(tr.trace.nodes.keys()))
    # print(list(pyro.get_param_store().keys()))
