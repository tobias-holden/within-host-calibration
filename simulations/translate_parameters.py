import os
import sys
import numpy as np
import pandas as pd

#### Read in Parameter Key
key_path = 'parameter_key.csv'
parameter_key = pd.read_csv(key_path)

distribution_names =["CONSTANT_DISTRIBUTION","UNIFORM_DISTRIBUTION",
                     "GAUSSIAN_DISTRIBUTION","EXPONENTIAL_DISTRIBUTION",
                     "LOG_NORMAL"]
#### Define Parameter Translator

def translate_parameters(key, guesses, ps_id):
    
    result = "Done" 
    MII_flag = False
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
    
    # Check IIVT Logic & do second translation on hyperparams
    ifrow = output[output['parameter'] == 'InnateImmuneDistributionFlag'].reset_index()
    #print(ifrow)
    iflag = ifrow['emod_value'][0]
    #print(iflag)
    #print(iflag)
    
    x1=output.loc[output['parameter'] == 'InnateImmuneDistribution1', 'emod_value'].item()
    x2=output.loc[output['parameter'] == 'InnateImmuneDistribution2', 'emod_value'].item()

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
    u = np.nan
    if loc['transform'][0]=='none':
        u = (value-loc['min'][0])/(loc['max'][0]-loc['min'][0])
        
    if loc['transform'][0]=='log':
        u = (np.log10(value) - np.log10(loc['min'])) / (np.log10(loc['max'])-np.log10(loc['min']))
    
    return(u)


if __name__ == '__main__':


    param_key=pd.read_csv("parameter_key.csv")
    
    print(emod_to_unit(param_key,"Anemia_Severe_Threshold",4.50775824973078))
    print(emod_to_unit(param_key,"Anemia_Severe_Inverse_Width",10))
    print(emod_to_unit(param_key,"Fever_Severe_Threshold",3.98354299722192))
    print(emod_to_unit(param_key,"Fever_Severe_Inverse_Width",27.5653580403806))
    print(emod_to_unit(param_key,"Parasite_Severe_Threshold",851031.287744526))
    print(emod_to_unit(param_key,"Parasite_Severe_Inverse_Width",56.5754896048744))
    
    # 
    # test_params= [0.235457679394, # Antigen switch rate (7.65E-10) 
    #               0.166666666667,  # Gametocyte sex ratio (0.2) 
    #               0.236120668037,  # Base gametocyte mosquito survival rate (0.00088) **
    #               0.394437557888,  # Base gametocyte production rate (0.0615)
    #               0.50171665944,   # Falciparum MSP variants (32)
    #               0.0750750750751, # Falciparum nonspecific types (76)
    #               0.704339142192,  # Falciparum PfEMP1 variants (1070)
    #               0.28653200892,   # Fever IRBC kill rate (1.4)
    #               0.584444444444,  # Gametocyte stage survival rate (0.5886)
    #               0.506803355556,  # MSP Merozoite Kill Fraction (0.511735)
    #               0.339794000867,  # Nonspecific antibody growth rate factor (0.5)  
    #               0.415099999415,  # Nonspecific Antigenicity Factor (0.4151) 
    #               0.492373751573,  # Pyrogenic threshold (15000)
    #               #1.0,            # Max Individual Infections (20)
    #               #0.666666666666,  # Erythropoesis Anemia Effect Size (3.5)
    #               #0.755555555555,  # RBC Destruction Multiplier (3.9)
    #               0.433677,         # Cytokine Gametocyte Inactivation (0.02)
    #               0.97,         # InnateImmuneDistributionFlag (Constant)
    #               0.25,         # Innate Immune Distribution hyperparameter
    #               0.3          # Innate Immune Distribution hyperparameter placeholder
    #              ]
    #              
    # tp=translate_parameters(param_key, test_params, 1)
    # #print(tp[['parameter','unit_value','emod_value']])
