In this branch, the configuration parameter `Innate_Immune_Variation_Type` is set to "CYTOKINE_KILLING".

- There **is** inter-individual variation applied to the Fever IRBC Kill Rate, but no age-effect for this parameter or Pyrogenic Threshold.
    - Pyrogenic Threshold is set at the given configuration parameter value for all individuals
    - Fever IRBC Kill Rate = **mod** $\times$ Fever IRBC Kill Rate

The value of the individual-level scale factor **mod** is drawn from the InnateImmuneDistribution defined in the demographics file. When included under calibration, new 'my_demographics_<>.json' files are generated with specified values for: 

| `InnateImmuneDistributionFlag` | Distribution | `InnateImmuneDistribution1` hyperparameter | `InnateImmuneDistribution2` hyperparameter |
|--------------------------------|--------------|--------------------------------------------|--------------------------------------------|
| 0                              | Constant     | NA                                         |  NA                                        |
| 1                              | Uniform      | min                                        |  max (1)                                   |
| 2                              | Exponential  | $\lambda$                                  |  NA                                        |
| 3                              | Gaussian     | $\mu$ (1)                                  |  $\sigma$                                  |
| 4                              | Log-Normal   | $\mu$ (0)                                  |  $\sigma$                                  |
