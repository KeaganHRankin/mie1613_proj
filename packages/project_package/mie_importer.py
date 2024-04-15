"""
This file contains a class for importing data for the simulation for MIE1613.
"""

# Import Libraries
# Data
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd



#####################################
class MIEImporter:
    """Importing simulation data."""
    def __init__(self):
        self.factor_range = 'likely'
        self.o_path = 'data/input_data/operational_intensity_2020_2050.csv'
        self.e_path = 'data/input_data/embodied_'+self.factor_range+'.csv'
        print("Importer created. Change self.factor_range for sensitivity analysis before importing.")
        
    def import_emission_data(self):
        """
        imports and formats the emboded and operational 
        emissions per unit data for each building type.
        factor_range = (min, likely, max) option for the embodied
        ghg emission factors of materials. Used for sensitivity analysis.
        year = year for embodied emissions data.
        """
        # Embodied Emissions - load
        embodied = pd.read_csv(self.e_path, index_col=0)

        # Operational Emissions - load pre-calculated table 2020-2050
        operational = pd.read_csv(self.o_path, index_col=0).loc[2023:]

        return (embodied, operational)