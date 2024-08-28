#### This repository contains input data and scripts used to perform regime calculations over California for the research paper by Shen et al., titled “Impact of fine particulate matter from wildfire smoke on near-surface O3 photochemistry”.

* “ROx_self_rxn_radical_change.ipynb”: This Jupyter Notebook selects all HOx self-reactions, calculates HOx radical loss coefficient for each reaction and saves the results to “Tropchem_ROx_self_rxn.txt”. 
*	“ROx_self_rxn_eq_num.ipynb”: This Jupyter Notebook generates the reaction index for each HOx self-reaction, combines it with radical change coefficients, and saves the combined data in “ROx_EQidx_RadicalChange.csv”.
*	“RONO2_rxn.ipynb”: This Jupyter Notebook generates the reaction index and calculates HOx radical loss for RO2 + NO = RONO2 reactions; the output is stored in “RONO2_EQidx_RadicalChange.csv”.
*	“regime_calculation.py”:  This script performs regime calculations, with outputs saved in the “data_output” directory.
