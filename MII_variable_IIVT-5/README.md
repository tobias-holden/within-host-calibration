In this branch, the configuration parameter `Innate_Immune_Variation_Type` is set to "PYROGENIC_THRESHOLD".

- There **is** inter-individual variation applied to the Pyrogenic Threshold, but no age-effect for this parameter on Fever IRBC Kill Rate.
    - Fever IRBC Kill Rate is set at the given configuration parameter value for all individuals
    - Pyrogenic Threshold = **mod** $\times$ Pyrogenic Threshold

The value of the individual-level scale factor **mod** is drawn from the InnateImmuneDistribution defined in the demographics file. When included under calibration, new 'my_demographics_<>.json' files are generated with specified values for: 

| `InnateImmuneDistributionFlag` | Distribution | `InnateImmuneDistribution1` hyperparameter | `InnateImmuneDistribution2` hyperparameter |
|--------------------------------|--------------|--------------------------------------------|--------------------------------------------|
| 0                              | Constant     | NA                                         |  NA                                        |
| 1                              | Uniform      | min                                        |  max (1)                                   |
| 2                              | Exponential  | $\lambda$                                  |  NA                                        |
| 3                              | Gaussian     | $\mu$ (1)                                  |  $\sigma$                                  |
| 4                              | Log-Normal   | $\mu$ (0)                                  |  $\sigma$                                  |
