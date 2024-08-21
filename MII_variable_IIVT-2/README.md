In this branch, the configuration parameter `Innate_Immune_Variation_Type` is set to "PYROGENIC_THRESHOLD_VS_AGE_INCREASING".

Based on Annie's synthesis of the article by

Isabel Rodriguez, Barraquer, Emmanuel Arinaitwe, Prasanna Jagannathan, Moses R Kamya, Phillip J Rosenthal, John Rek, Grant Dorsey, Joaniter Nankabirwa, Sarah G Staedke, Maxwell Kilama, Chris Drakeley, Isaac Ssewanyana, David L Smith, Bryan Greenhouse (2018) 
*"Quantification of anti-parasite and anti-disease immunity to malaria as a function of age and exposure"* eLife 7:e35832

- There is no inter-individual variation applied to either the Pyrogenic_Threshold or the Fever_IRBC_Kill_Rate, except by age.
    - All individuals of the same age have the same value for each parameter.
    - Fever_IRBC_Kill_Rate is set at the given configuration parameter value.
    - The given configuration parameter value for Pyrogenic threshold is a minimum... Pyrogenic threshold increases with age from birth, to a ceiling (currently set at 100,000).
        - Pyrogenic Threshold = $$min($$ Pyrogenic Threshold, $$10^{(0.132 \times Age)}, 100000)$$ 
 
- The default demographics files *demographics_cohort_1000.json* and *demographics_vital_1000.json* are used.


