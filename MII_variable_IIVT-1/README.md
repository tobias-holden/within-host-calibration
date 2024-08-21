In this branch, the configuration parameter `Innate_Immune_Variation_Type` is set to "PYROGENIC_THRESHOLD_VS_AGE".

- There is no inter-individual variation applied to either the Pyrogenic_Threshold or the Fever_IRBC_Kill_Rate, except by age.
    - All individuals of the same age have the same value for each parameter.
    - Fever_IRBC_Kill_Rate is set at the given configuration parameter value.
    - The given configuration parameter value for Pyrogenic threshold is a maximum.
        - For Age < 2 years: Pyrogenic Threshold = Pyrogenic Threshold $+ 0.035 \times$ Pyrogenic Threshold $\times Age$
        - For Age ≥ 2 years: Pyrogenic Threshold = Pyrogenic Threshold $\times 0.965e^{-0.9(Age-2)} +$ Pyrogenic Threshold $\times 0.1$
 
- The default demographics files *demographics_cohort_1000.json* and *demographics_vital_1000.json* are used.
