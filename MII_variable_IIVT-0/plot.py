import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol
import torch
from botorch.acquisition.objective import IdentityMCObjective

#convert plot_x_flat y axis
from simulations.translate_parameters import translate_parameters

def plot_MSE(bo, n_init=0):
    n = bo.Y.shape[0]-n_init
    
    print(n)
    
    print(len(bo.iterations[-n:].cpu().numpy()))
    print(len(bo.objective(bo.Y)[-n:].squeeze().cpu().numpy()))
    print(len(bo.objective(bo.Y_pred_mean)[-n:].squeeze().cpu().numpy()))

    df = pd.DataFrame.from_dict({
        'iteration': bo.iterations[-n:].cpu().numpy(),
        'Y': bo.objective(bo.Y)[-n:].squeeze().cpu().numpy(),
        'Y_mean': bo.objective(bo.Y_pred_mean)[-n:].squeeze().cpu().numpy()
    })
    
    df['squared_error'] = (df.Y - df.Y_mean)**2
    df_mean = df.groupby('iteration').mean().reset_index()

    plt.figure()
    plt.plot(df_mean['iteration'], df_mean['squared_error'], label="Mean Squared Error")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Squared Error")

def plot_runtimes(bo):
    iterations = np.arange(len(bo.timer_model))
    model_runtimes = bo.timer_model.squeeze().cpu().numpy()
    batch_generator_runtimes = bo.timer_batch_generator.squeeze().cpu().numpy()
    
    plt.figure()
    plt.plot(iterations, model_runtimes, label="Model Runtime")
    plt.plot(iterations, batch_generator_runtimes, label="Batch Generator Runtime")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Runtime")

def plot_convergence(bo, ymin = None, ymax = None, negate=False, title="", objective = IdentityMCObjective()):
    Y = bo.objective(bo.Y)
    
    sign = (-1.0 if negate else 1.0)

    if ymin is None:
        ymin = Y.min().cpu().numpy()

    if ymax is None:
        ymax = Y.max().cpu().numpy()
    
    if negate:
        temp = ymin
        ymin = ymax * sign
        ymax = temp * sign

    n = len(Y)

    fig = plt.figure()
    ax = fig.add_subplot()

    oY = []
    max_y = -np.Infinity
    Ymax = []

    for y in Y.cpu().numpy():
        max_y = max(max_y, y)
        Ymax.append(max_y)
        oY.append(y)
    Ymax = np.array(Ymax)

    oY = np.array(oY)

    ax.scatter(np.arange(0, n), oY * sign, color='black', linewidth=1, marker='o', label="Sample")
    ax.plot(np.arange(0, n), Ymax * sign, color='red', linewidth=2, marker='', markevery=1, markersize=2, label="Best value")


    ax.set_yscale('log')
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(loc="upper right")


def plot_space(bo, ymin, ymax, labels, n=1, m=1):
    m += 1

    Xprior = bo.X
    Yprior = bo.Y

    #Yprior = (Yprior - Yprior.mean()) / Yprior.std()

    test_X = torch.linspace(0, 1, 1001, dtype=bo.dtype, device=bo.device)

    print("Fitting model for space analysis...")
    model = bo.model
    model.fit(Xprior, Yprior)
    print("Done.")

    with torch.no_grad():
        xopt = Xprior[Yprior.argmax()]
        yopt = bo.objective(Yprior).max().detach().cpu().numpy()

        popt = bo.objective(model.posterior(xopt.squeeze().unsqueeze(0)).mean).cpu().numpy() #rsample().squeeze().detach().cpu().numpy()

        print("Opt =", yopt, "(predicted", popt, ") at eval", Yprior.argmax(), "out of", len(Yprior), "evals.")

        fig = plt.figure(figsize=(20,10))

        for i in range(bo.X.shape[-1]+1):
            if i<bo.X.shape[-1]:
                opt_x = xopt.detach().clone().cpu().numpy()
                test_x = torch.tensor([opt_x for i in range(len(test_X))], dtype=bo.dtype, device=bo.device)
                test_x[:,i] = test_X
                
                test_posterior = model.posterior(test_x)
                test_Y = bo.objective(test_posterior.mean).cpu().numpy() #rsample().squeeze().detach().cpu().numpy()

                SD = torch.sqrt(bo.objective(test_posterior.variance)).numpy()
                lower = test_Y - 1.96 * SD
                upper = test_Y + 1.96 * SD

                pred_posterior = model.posterior(Xprior)
                pred_Y = bo.objective(pred_posterior.mean).squeeze().cpu().numpy()

                ax = fig.add_subplot(n,m,i+1)

                ax.scatter(Xprior[:,i].squeeze().cpu().numpy(), bo.objective(Yprior).squeeze().detach().cpu().numpy(), lw=0, s=8.0, color="blue") #, label="Simulated")
                ax.scatter(Xprior[:,i].squeeze().cpu().numpy(), pred_Y, lw=0, s=8.0, color="orange", alpha=0.5) #, label="Predicted")
                
                ax.plot(test_X.squeeze(), test_Y, color="orange", label=labels[i])
                ax.fill_between(test_X.cpu().numpy(), lower, upper, alpha=0.5) #, label="Confidence")
                ax.scatter(xopt[i].cpu().numpy(), yopt, s=24.0, color="red") #, label="Simulated Optimal")

                ax.set_ylim(ymax, ymin)
                ax.set_xlim(0, 1)
                # ax.set_yscale('symlog')
                ax.legend(loc="lower left")
            else:
                ax = fig.add_subplot(n,m,i+1)
                ax.scatter([1], [1], lw=0, s=8.0, color="blue", label="Simulated")
                ax.scatter([1], [1], lw=0, s=8.0, color="orange", alpha=0.5, label="Predicted")
                ax.plot([1], [1], color="orange", label="Mean")
                ax.fill_between([1], [1], alpha=0.5, label="Confidence")
                ax.scatter([1], [1], s=24.0, color="red", label="Simulated Optimal")
                ax.set_ylim(-30, -200)
                ax.set_xlim(0, 1)
                ax.plot(1, 1)
                ax.legend(loc="lower left")
        plt.tight_layout()

def plot_prediction_error(bo):
    """Plot of posterior mean vs y, optional test data"""
    fig = plt.figure()
    
    ax = fig.add_subplot(1,1,1)

    X = bo.X.cpu().numpy()
    Y = bo.objective(bo.Y).cpu().numpy()
    Y_pred = bo.objective(bo.Y_pred_mean).cpu().numpy()
    n = len(Y)-1
    print(n)
    print(len(Y_pred))
    ax.scatter(np.arange(n), -Y[-n:], s=8, label="Samples")
    ax.scatter(np.arange(n), -Y_pred[-n:], s=8,label="Predictions")

    ax.plot(np.arange(n), abs(Y[-n:]-Y_pred[-n:]), color='red', label="Absolute Error")

    ax.legend(loc="upper right")

def plot_y_vs_posterior_mean(
        bo,
        log = False,
        top = False,
        X_test = None,
        Y_test = None,
        n_init = 0
    ):
        """Plot of posterior mean vs y, optional test data"""

        plt.figure()
        
        X = bo.X
        Y = bo.Y
        Y_pred = bo.Y_pred_mean

        model = bo.model
        model.fit(X, Y)

        n = len(bo.Y)-n_init

        Y_pred = Y_pred[-n:].cpu().numpy()
        Y = Y[-n:].cpu().numpy()

        if X_test is not None:
            Y_pred_test = model.posterior_mean(X_test).cpu().numpy()
            Y_pred = torch.vstack([Y_pred, pp_test])
            Y = torch.vstack([Y, Y_test])

        all_vals = np.concatenate([Y, Y_pred])
        plt.scatter(Y, Y_pred)

        ymin = np.inf
        ymax = -np.inf
        for y in Y:
            if y < ymin and not np.isnan(y):
                ymin = y
            if y > ymax and not np.isnan(y):
                ymax = y

        plt.plot([ymin, ymax], [ymin, ymax], c = 'k')
        
        xlabel = "$Y$"
        ylabel = "$Y_pred$"
        if log:
            xlabel = xlabel + " (log)"
            ylabel = ylabel + " (log)"
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

def plot_X_flat(bo, skip = 0, negate=True, param_key = None, labels = None):
    X = bo.X.cpu().numpy()
    Y = bo.objective(bo.Y).cpu().numpy()
    fig = plt.figure(figsize=(29,30))

    besty = []
    bestx = []
    besti = []
    best = -np.Infinity
    
    converted_X = []
    for i in range(skip, len(X)):
        y = Y[i]
        converted_X.append(translate_parameters(param_key,X[i],0)["emod_value"])
        if y > best:
            best = y
            bestx.append(translate_parameters(param_key,X[i],0)["emod_value"])
            besty.append(y)
            besti.append(i)
    besty = np.array(besty)
    bestx = np.array(bestx)
    converted_X = np.array(converted_X)

    for i in range(X.shape[-1]):
        ax = fig.add_subplot(6,4,i+1)
        #sign = (-1.0 if negate else 1.0)
        x = np.arange(skip,len(X))
        ax.scatter(x,converted_X[skip:,i], c=Y[skip:], marker='o', s=16, linewidth=1.0, label=labels[i] if labels is not None else str(i), cmap=plt.cm.viridis)
        ax.plot(besti,bestx[:,i], marker='^', color='red', linewidth=1.0, markersize=1.0, label=labels[i] if labels is not None else str(i))
        if param_key['transform'][i] is not None:
            if param_key['transform'][i] == 'log':
                ax.set_yscale('log', base=2)
            
        #best_ll = (besti[-1], bestx[-1,i])
        #bestx_value = bestx[-1,i]
        #if type(bestx_value) == float:
        #    bestx_value = round(bestx_value,3)
        #bestx_value = str(bestx_value)
        #ax.annotate(bestx_value, best_ll)
        imax = Y.argmax()
       ## ax.set_ylim(0, 1)
        ax.legend(loc="upper left")

def plot_length_scales(bo):
    df = pd.DataFrame(bo.length_scales).add_prefix("X").reset_index().melt(id_vars="index")
    g = sns.FacetGrid(data=df, col='variable', col_wrap=5)
    g.map_dataframe(sns.lineplot, x='index', y='value')
    
    
def sobol_indices(bo, nSamples = 1000):
    Xprior = bo.X
    Yprior = bo.Y

    model = bo.model
    model.fit(priorX, Yprior)

    problem = {
        'num_vars': Xprior.shape[1],
        'names': [str(i) for i in range(Xprior.shape[1])],
        'bounds': [[0.0, 1.0] for i in range(Xprior.shape[1])]
    }

    print(Xprior.shape[1])
    X = saltelli.sample(problem, nSamples)

    S1 = np.zeros((12,23))
        
    model.fit(Xprior, Yprior)

    Y = np.zeros(len(X))

    for i, x in enumerate(X):
        x = torch.tensor(x).to(Xprior).unsqueeze(0)
        with torch.no_grad():
            Y[i] = model.posterior(x).mean.cpu().numpy()

    return sobol.analyze(problem, Y)
    
