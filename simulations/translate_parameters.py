import os
import sys
import numpy as np
import pandas as pd
import itertools
#### Read in Parameter Key
key_path = 'parameter_key.csv'
parameter_key = pd.read_csv(key_path)

distribution_names =["CONSTANT_DISTRIBUTION","UNIFORM_DISTRIBUTION",
                     "GAUSSIAN_DISTRIBUTION","EXPONENTIAL_DISTRIBUTION",
                     "LOG_NORMAL","BIMODAL_DISTRIBUTION"]
#### Define Parameter Translator

def translate_parameters(key, guesses, ps_id):
    
    result = "Done" 
    MII_flag = False
    LTD_flag = False
    AIK_flag = False
    output = pd.DataFrame({"parameter": [], 
                           "param_set": [],
                           "team_default":[],
                           "unit_value": [],
                           "min": [],
                           "max": [],
                           "transformation": [],
                           "type":[],
                           "emod_value": []}, columns=['parameter',"param_set",'unit_value', 'emod_value',"min","max","team_default","transformation","type"])
    if(len(guesses) % len(key.index) != 0):
        result="Error: Number of guesses must equal the number of parameters in the key"
        print(result,"(",len(key.index),")")
        return
    #if(any(guesses<0) or any(guesses>1)):
     #   result="Error: Guesses must lie between 0 and 1"
      #  print(result)
       # return
    
    for index, row in key.iterrows():
        # Scale to parameter range
        MII_flag=False
        LTD_flag=False
        AIK_flag=False
        # Scale to parameter range
        value = row['min']+guesses[index]*(row['max']-row['min'])
        #print(f"{guesses[index]} --> {value}")
        # Apply Transformations
        if(row['transform']=='log'):
            value = np.log10(row['min'])+guesses[index]*(np.log10(row['max'])-np.log10(row['min']))
            #print(f"Transforming: {value} --> {10**value}")
            value = 10**(value)
            #print(f"{value} --> {10**value}")
            # value = row['max'] * guesses[index]**2  # 5: with range of 0-10^3, values up to 0.25 will shrink (translate to <1), above = grow
            #                                         # 10: shifts inflection point to 0.5
            #                                         # 2: shifts inflection point to < 0.05 
        elif(row['transform']=='IIVT'):
            if(guesses[index] <= float(1/3)):#float(0.5)):#
                value = "NONE"
            elif(guesses[index]<= float(2/3)):
                value = "PYROGENIC_THRESHOLD_VS_AGE"
            else: 
                value = "PYROGENIC_THRESHOLD_VS_AGE_INC"
        
        # Restrict Min/Max
        if(row['transform'] == 'log'):
            if(value < 10**np.log10(row['min'])):
                value = 10**np.log10(row['min'])
            if(value > 10**np.log10(row['max'])):
                value = 10**np.log10(row['max'])
        elif(row['transform'] != 'IIVT'):
            if(value < row['min']):
                value = row['min']
            if(value > row['max']):
                value = row['max']
        
        
        default = row['team_default']
        if (default==''):
            default = row['emod_default']
            
        if(row['parameter_name']=='Max_Individual_Infections'):
            if(guesses[index]==-1):
              MII_flag=True
              
        if(row['parameter_name']=="Antibody_Days_To_Long_Term_Decay"):
            if(guesses[index]==-1):
              LTD_flag=True
        
        if(row['parameter_name']=="Antibody_IRBC_Kill_Rate"):
            if(guesses[index]==-1):
              AIK_flag=True
        
        # Convert Data Types
        if(row['type'] == 'integer'):
            value = np.trunc(value)
        if(row['parameter_name']=="InnateImmuneDistributionFlag"):
            value=np.trunc(value)
            value=distribution_names[int(value)]
            
        
        default = row['team_default']
        if (default==''):
            default = row['emod_default']
        
        # Fix initial MII
        if MII_flag: 
            value=np.trunc(3.0)
        # Fix default LTD (never)
        if LTD_flag:
            value=np.trunc(365000)
        if AIK_flag:
            value=1.596
        
        new_row = pd.DataFrame({"parameter": [row['parameter_name']], 
                                "param_set": [ps_id],
                                "team_default":[default],
                                "unit_value": [guesses[index]],
                                "min": [row['min']],
                                "max": [row['max']],
                                "transformation": [row['transform']],
                                "type":[row['type']],
                                "emod_value": [value]}, columns=['parameter',"param_set",'unit_value', 'emod_value',"min","max","team_default","transformation","type"])
        
        output = pd.concat([output, new_row])
        
        #print(row['parameter_name'])
        #print(guesses[index],'-->',value)
    
    output = output.reset_index(drop=True)
    
    iflag="BIMODAL_DISTRIBUTION"
    
    # Check IIVT Logic & do second translation on hyperparams
    #ifrow = output[output['parameter'] == 'InnateImmuneDistributionFlag'].reset_index()
    #print(ifrow)
    #iflag =ifrow['emod_value'][0]
    #print(iflag)
    #print(iflag)
    
    #x1=output.loc[output['parameter'] == 'InnateImmuneDistribution1', 'emod_value'].item()
    x2=output.loc[output['parameter'] == 'InnateImmuneDistribution2', 'emod_value'].item()
    
    iflag=="BIMODAL_DISTRIBUTION"

    if iflag=='CONSTANT_DISTRIBUTION':
      #print("CONSTANT")
      output.loc[output['parameter'] == 'InnateImmuneDistribution1', 'emod_value'] = 0.0
      output.loc[output['parameter'] == 'InnateImmuneDistribution2', 'emod_value'] = 0.0
    elif iflag=='UNIFORM_DISTRIBUTION':
      #print("UNIFORM")
      # Min
      output.loc[output['parameter'] == 'InnateImmuneDistribution1', 'emod_value'] = 1.0-x1
      # Max
      output.loc[output['parameter'] == 'InnateImmuneDistribution2', 'emod_value'] = 1.0 + x1
    elif iflag=='GAUSSIAN_DISTRIBUTION':
      #print("GAUSSIAN")
      # Mean fixed at 1.0
      output.loc[output['parameter'] == 'InnateImmuneDistribution1', 'emod_value'] = 1.0
      # Convert standard deviation
      output.loc[output['parameter'] == 'InnateImmuneDistribution2', 'emod_value'] = x1
    elif iflag=='EXPONENTIAL_DISTRIBUTION':
      #print("EXPONENTIAL")
      output.loc[output['parameter'] == 'InnateImmuneDistribution1', 'emod_value'] = x1
      # No second parameter, set to zero.
      output.loc[output['parameter'] == 'InnateImmuneDistribution2', 'emod_value'] = 0.0
    elif iflag=='LOG_NORMAL':
      #print("LOGNORMAL")
      # Fix mean at 0.0
      output.loc[output['parameter'] == 'InnateImmuneDistribution1', 'emod_value'] = 0.0
      # standard deviation
      output.loc[output['parameter'] == 'InnateImmuneDistribution2', 'emod_value'] = x1
    elif iflag=='BIMODAL_DISTRIBUTION':
      #output.loc[output['parameter'] == 'InnateImmuneDistribution1', 'emod_value'] = 0.6
      # relative scale
      output.loc[output['parameter'] == 'InnateImmuneDistribution2', 'emod_value'] = x2
      
     
    return(output)

#### Generate Initial Samples

def get_initial_samples(key, size=1):
    n_param = len(key.index)
    values = np.linspace(0,1,size)
    np.random.shuffle(values)
    values = list(values)
    for i in range(n_param-1):
        x=np.linspace(0,1,size)
        np.random.shuffle(x)
        x=list(x)
        values.extend(x)

    values = np.array(values).reshape(n_param,size).transpose()
    
    #values['param_set'] = np.trunc(values.index/15)+1
    #print(values.shape)
    #print(values)
    return(values)

def emod_to_unit(key,param,value): 
    loc = key[key['parameter_name']==param].reset_index()
    #print(loc)
    u = np.nan
    if loc['transform'][0]=='none':
        u = (value-loc['min'][0])/(loc['max'][0]-loc['min'][0])
        
    if loc['transform'][0]=='log':
        u = (np.log10(value) - np.log10(loc['min'])) / (np.log10(loc['max'])-np.log10(loc['min']))
    
    return(u)

def best_emod_to_unit(key,param,value): 
    loc = key[key['parameter']==param].reset_index()
    #print(loc)
    u = np.nan
    if loc['transformation'][0]=='none':
        u = (value-loc['min'][0])/(loc['max'][0]-loc['min'][0])
        
    if loc['transformation'][0]=='log':
        u = (np.log10(value) - np.log10(loc['min'])) / (np.log10(loc['max'])-np.log10(loc['min']))
    
    return(u)

if __name__ == '__main__':


    param_key=pd.read_csv("parameter_key.csv")
    for index, row in param_key.iterrows():
        print(f'{row["parameter_name"]} of {row["team_default"]}')
        print(emod_to_unit(param_key,row["parameter_name"],row["team_default"]))
        
    # param_key=pd.read_csv("output/250218_test_phase0/LF_12/emod.best.csv")
    # for index, row in param_key.iterrows():
    #     print(f'{row["parameter"]} of {row["emod_value"]}')
    #     print(best_emod_to_unit(param_key,row["parameter"],row["emod_value"]))
