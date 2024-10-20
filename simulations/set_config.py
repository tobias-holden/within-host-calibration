def set_config( config, tmp_loc = [], rate= 1.0, infectivity = 1.0 ):
    config.parameters.Simulation_Type = "MALARIA_SIM" 
    config.parameters.Acquisition_Blocking_Immunity_Decay_Rate = 0.01
    config.parameters.Acquisition_Blocking_Immunity_Duration_Before_Decay = 90
    config.parameters.Age_Initialization_Distribution_Type = 'DISTRIBUTION_SIMPLE'

    config.parameters.Incubation_Period_Exponential = 233.33
    config.parameters.Individual_Sampling_Type = "TRACK_ALL"

    config.parameters.Infectious_Period_Constant = 0
    config.parameters.Enable_Birth = 1
    # config.parameters.Enable_Coinfection = 1
    config.parameters.Enable_Demographics_Birth = 1
    config.parameters.
    config.parameters.Enable_Demographics_Reporting = 0
    # config.parameters.Enable_Immune_Decay = 0
    config.parameters.Migration_Model = "NO_MIGRATION"
    config.parameters.Run_Number = 99 
    config.parameters.Simulation_Duration = 70*365+1
    config.parameters.Enable_Demographics_Risk = 1
    config.parameters.Enable_Maternal_Infection_Transmission = 0
    config.parameters.Enable_Natural_Mortality = 1

    config.parameters.Maternal_Antibodies_Type = "CONSTANT_INITIAL_IMMUNITY"

    return config
