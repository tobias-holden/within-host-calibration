from run_sims import submit_sim
from run_analyzers import run_analyzers
from get_eradication import get_eradication
from compare_to_data.run_full_comparison import compute_LL_across_all_sites_and_metrics
import argparse
import params as params
import manifest as manifest
import os, sys, shutil
import time
import matplotlib.pyplot as plt
from load_inputs import load_sites
import pandas as pd
import numpy as np
from helpers import load_coordinator_df
from botorch.utils.transforms import unnormalize
import torch

from gpytorch.constraints import Interval, GreaterThan, LessThan

from translate_parameters import translate_parameters, get_initial_samples

sites = load_sites()

def my_func(X,wdir,JS=False):
  # Supply parameters to X
  get_eradication(manifest.use_local_eradication)
  param_key=pd.read_csv("parameter_key.csv")
  df = pd.DataFrame({'parameter':[], 'unit_value': [], 'emod_value':[], 'type':[], 'param_set':[]})
  i=1
  print("translating parameters...", flush=True)
  for x in X:
      a = translate_parameters(param_key,x,i)
      #a['param_set'] = np.repeat(i,len(a))
      i=i+1
      df = pd.concat([df,a])
  df.to_csv(f"{wdir}/translated_params.csv")
 
  ## Commented out to jump-start - NEED TO UNCOMMENT
  print("submitting simulations...", flush=True)

  if not JS:
      for my_site in sites:
          if os.path.exists(os.path.join(manifest.simulation_output_filepath,my_site)):
              shutil.rmtree(os.path.join(manifest.simulation_output_filepath,my_site))
    
          coord_df=load_coordinator_df()
          ns = coord_df.at[my_site, 'nSims']
          submit_sim(site=my_site, X=df, nSims=ns)
    
  #for my_site in sites:
   #   run_analyzers(site=my_site)
  
  print("waiting for outputs...", flush=True)
  while True:
      outputs = []
      for my_site in sites:
          outputs.append(os.path.exists(os.path.join(manifest.simulation_output_filepath,my_site,'finished.txt')))
      if all(outputs):#os.path.exists(manifest.simulation_output_filepath,my_site): 
          print("Computing metrics...", flush=True)
          Y = compute_LL_across_all_sites_and_metrics(numOf_param_sets=len(X))
          break
      time.sleep(120)   
    
  return(Y)


def my_func2(X,wdir):
  # Supply parameters to X
  get_eradication(manifest.use_local_eradication)
  param_key=pd.read_csv("parameter_key.csv")
  df = pd.DataFrame({'parameter':[], 'unit_value': [], 'emod_value':[], 'type':[], 'param_set':[]})
  i=1
  for x in X:
      a = translate_parameters(param_key,x,i)
      #a['param_set'] = np.repeat(i,len(a))
      df = pd.concat([df,a])
      #x=torch.tensor(x)
      y1 = x[0]**2 
      y2 = x[1] 
      y3 = -(x[2]**2) 
      y4 = x[3] - (2*x[4])
      y5 = y1
      y6 = 2*y2
      y7 = y1 - y3
      y8 = y2
      y9 = y4 + y5
      y10 = 2*y2 - y3
      y11 = y3
      y12 = y4+y1
      y14 = y6 - y3
      y15 = y5
      y16 = y3
      y17 = y6 + y4
      y18 = y2
      y19 = y1 + y2 - y4 + y5
      y20 = y6 - y2
      y21 = y3 - y2
      y22 = y4 + 2*y1 + 3*y3
      y23 = y5 - y7**2
      y24 = y8 + y4
      y25 = y2**2 + y1**3
      y26 = y4 + y3 - y2 - y5
      y= [y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,
          y15,y16,y17,y18,y19,y20,y21,y22,y23,y24,y25,y26]
      if i == 1:
          Y = y
      else:
          Y = torch.stack((Y,y))
      i=i+1

  df.to_csv(f"{wdir}/translated_params.csv")
  return(Y)


if __name__ == '__main__':
    param_key=pd.read_csv("parameter_key.csv")
    X = get_initial_samples(param_key, 100)
    X = X[1:5]
    my_func(X)
