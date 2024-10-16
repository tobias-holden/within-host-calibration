In this branch, the configuration parameter `Innate_Immune_Variation_Type` is set to "PYROGENIC_THRESHOLD_VS_AGE_IIV".

- There **is** inter-individual variation applied to the Pyrogenic_Threshold, and an age-effect.
    - Fever_IRBC_Kill_Rate is set at the given configuration parameter value.
    - The given configuration parameter value for Pyrogenic threshold is a maximum, and decreases with age.
        - For Age < 2 years: Pyrogenic Threshold = **mod** $\times$ Pyrogenic Threshold $+ 0.035 \times$ Pyrogenic Threshold $\times Age$
        - For Age ≥ 2 years: Pyrogenic Threshold = **mod** $\times$ Pyrogenic Threshold $\times 0.965e^{-0.9(Age-2)} +$ Pyrogenic Threshold $\times 0.1$


The value of the individual-level scale factor **mod** is drawn from the InnateImmuneDistribution defined in the demographics file. When included under calibration, new 'my_demographics_<>.json' files are generated with specified values for: 

| `InnateImmuneDistributionFlag` | Distribution | `InnateImmuneDistribution1` hyperparameter | `InnateImmuneDistribution2` hyperparameter |
|--------------------------------|--------------|--------------------------------------------|--------------------------------------------|
| 0                              | Constant     | NA                                         |  NA                                        |
| 1                              | Uniform      | min                                        |  max (1)                                   |
| 2                              | Exponential  | $\lambda$                                  |  NA                                        |
| 3                              | Gaussian     | $\mu$ (1)                                  |  $\sigma$                                  |
| 4                              | Log-Normal   | $\mu$ (0)                                  |  $\sigma$                                  |

