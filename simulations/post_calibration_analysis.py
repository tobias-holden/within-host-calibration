import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import sys
from post_calibration_GP import fit_GP_to_objective, get_script_path
import argparse
  
herepath = get_script_path()

def plot_predictions(experiment='',exclude_count=0):
    Y = torch.load(os.path.join(herepath,'output',experiment,'Y.pt'))
    Y_pred_mean = torch.load(os.path.join(herepath,'output',experiment,'Y_pred_mean.pt'))
    Y_pred_var = torch.load(os.path.join(herepath,'output',experiment,'Y_pred_var.pt'))
    # Convert tensors to NumPy arrays
    Y = Y.numpy().flatten()
    mean = Y_pred_mean.numpy().flatten()
    var = Y_pred_var.numpy().flatten()
    n_missing=len(mean)-len(Y)
    if n_missing>0:
        Y=np.append(Y,np.repeat(0.1,n_missing))
    if len(mean) > exclude_count:
        Y = Y[exclude_count:]
        mean = mean[exclude_count:]
        var = var[exclude_count:]
        # Calculate standard deviation
        std_dev = np.sqrt(var)
        # Define confidence interval (e.g., 95% confidence interval)
        confidence_interval = 1.96 * std_dev
        # Define x values (e.g., indices or other x-axis values)
        x = np.arange(len(mean))
        # Plot the mean predictions
        plt.figure(figsize=(10, 6))
        plt.plot(x, mean, label='Predicted Mean', color='blue')
        # Plot the confidence interval
        #plt.fill_between(x, mean - confidence_interval, mean + confidence_interval, color='blue', alpha=0.2, label='95% Confidence Interval')
        plt.plot(x,Y,label='Observation',marker='o',color='blue',linestyle='')
        # Add labels and title
        plt.xlabel('Index')
        plt.ylabel('Predicted Value')
        plt.title('Predictions with Confidence Interval')
        plt.legend()
        # Show the plot
        plt.show()
        plt.savefig(os.path.join(herepath,'output',experiment,"performance/GP/predictions.png"), dpi=300)  # Save as PNG with high resolution
        plt.savefig(os.path.join(herepath,'output',experiment,"performance/GP/predictions.pdf"))  
    else:
        print("The tensor length is less than or equal to the number of values to exclude.")
    return

def plot_timers(experiment=''):
    timer_model = torch.load(os.path.join(herepath,'output',experiment,'timer_model.pt'))
    timer_batch_generator = torch.load(os.path.join(herepath,'output',experiment,'timer_batch_generator.pt'))

    timer_model = timer_model.numpy().flatten()
    timer_batch_generator = timer_batch_generator.numpy().flatten()
    # Define x values (e.g., indices or other x-axis values)
    x = np.arange(len(timer_model))
    
    # Plot the mean predictions
    plt.figure(figsize=(10, 6))
    plt.plot(x, timer_model, label='Model Fitting', color='blue')
    plt.plot(x, timer_batch_generator, label='Posterior Sampling', color='red')
    # Add labels and title
    plt.xlabel('Batch')
    plt.ylabel('Time (sec)')
    plt.title('Timing')
    plt.legend()
    # Show the plot
    plt.show()
    plt.savefig(os.path.join(herepath,'output',experiment,"performance/GP/timing.png"), dpi=300)  # Save as PNG with high resolution
    plt.savefig(os.path.join(herepath,'output',experiment,"performance/GP/timing.pdf")) 
    
    return

def plot_length_scales(experiment=''):
    length_scales = torch.load(os.path.join(herepath,'output',experiment,'length_scales.pt'))
    length_scales = torch.transpose(length_scales,0,1)
    nparam=list(length_scales.shape)[0]
    rounds=list(length_scales.shape)[1]
    parameter_names = pd.read_csv(os.path.join(herepath,'parameter_key.csv'))
    parameter_names = parameter_names['parameter_label'].tolist()
    num_plots = nparam
    cols = int(np.ceil(np.sqrt(num_plots)))  # Number of columns in the grid
    rows = int(np.ceil(num_plots / cols)) 
    fig, axes = plt.subplots(rows, cols, figsize=(15, 2 * rows), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten the 2D array of axes to make indexing easier
    # Plot each row of the tensor
    x_values = torch.arange(rounds).tolist()
    for i in range(num_plots):
        axes[i].plot(x_values, length_scales[i].numpy(), marker='', linestyle='-')
        axes[i].set_title(f'{parameter_names[i]}')
        axes[i].set_ylabel('Length Scale')
        axes[i].set_xlabel('Batch')
    # Hide any unused subplots
    for j in range(num_plots, len(axes)):
        axes[j].axis('off')
    # Adjust layout
    plt.tight_layout()
    plt.show()
    # Save the plot as PNG and PDF
    plt.savefig(os.path.join(herepath,'output',experiment,"performance/GP/length_scales.png"), dpi=300)  # Save as PNG with high resolution
    plt.savefig(os.path.join(herepath,'output',experiment,"performance/GP/length_scales.pdf"))  
    return

def post_calibration_analysis(experiment='',length_scales_by_objective=True,length_scales_plot=True,prediction_plot=True,exclude_count=0,timer_plot=True):
                                
    os.makedirs("/".join((herepath,"output",experiment,"performance","GP")),exist_ok=True)
    print(" ".join(("Loading botorch objects for experiment",experiment)))
    
    if prediction_plot:
        plot_predictions(experiment,exclude_count)
    if timer_plot:
        plot_timers(experiment)
    if length_scales_plot:
        plot_length_scales(experiment)
    if length_scales_by_objective:
        # Fit single-task GP  to all site-metric objectives in all_scores.csv
        scores = pd.read_csv(os.path.join(herepath,'output',experiment,'all_LL.csv'))
        # skip no_blood objectives
        filtered_scores = scores[scores['metric'] != 'no_blood']
        # Get unique combinations of 'site' and 'metric' from scores dataframe
        unique_combinations = filtered_scores[['site', 'metric']].drop_duplicates()
        # Iterate over unique combinations
        for _, row in unique_combinations.iterrows():
            site = row['site']
            metric = row['metric']
            # Fit single task GP to single site-metric objective
            fit_GP_to_objective(exp=experiment,site=site,metric=metric)
    return
   

if __name__=="__main__":
  
      # Create the parser
    parser = argparse.ArgumentParser(description='Post Calibration Analysis Parameters')

    # Add arguments
    parser.add_argument('--experiment', type=str, required=True, help='Experiment label')
    parser.add_argument('--length_scales_by_objective', default=False, action='store_true', 
                        help='Whether to calculate length scales by objective')
    parser.add_argument('--length_scales_plot', default=False,action='store_true', 
                        help='Whether to plot length scales')
    parser.add_argument('--prediction_plot', default=False,action='store_true',  
                        help='Whether to plot predictions')
    parser.add_argument('--exclude_count', type=int, default=1000, 
                        help='Count to exclude from analysis')
    parser.add_argument('--timer_plot', default=False,action='store_true',  
                        help='Whether to plot timers')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    post_calibration_analysis(
        experiment=args.experiment,
        length_scales_by_objective=args.length_scales_by_objective,
        length_scales_plot=args.length_scales_plot,
        prediction_plot=args.prediction_plot,
        exclude_count=args.exclude_count,
        timer_plot=args.timer_plot
    )
