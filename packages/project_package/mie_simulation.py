"""
This file contains a class for running the simulation for MIE1613.
"""

# Import Libraries
# Data
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd


#####################################

class MIESimulation:
    """Running the simulation."""
    def __init__(self):
        # random seed
        self.seed = 1614

        # year for single period simulation
        self.year = 2023

        # distribution for embodied emissions
        self.e_distribution = stats.lognorm

        # lmh name and percentages
        # nlmh can be modified, lmh should be left as is (used in many functions).
        self.lmh = ['Single Family','Missing Middle','Mid High Rise']
        self.nlmh = [0.09, 0.24, 0.67]

        # input data - embodied and operational
        self.e_data = 0
        self.o_data = 0

        print("Sim object created. Initialize self.e_data and self.o_data")


    #####################################
    # Input function(s)
    #####################################
    def embodied_fitter_mle(self, data, distribution=stats.lognorm):
        """
        Fit data to distribution
        using maxmimum likelyhood estimation.
        distribution = scipy type dist to fit.
        data = imported embodied emissions data from import_emission_data().

        fits the three major housing types, returns parameters.
        """
        # loop and fit each
        parameters = {}
        for house_type in self.lmh:
            d_col = 'ghg_per_unit'

            data_i = data.loc[data['labels_general']==house_type][d_col]
            parameters[house_type] = distribution.fit(data_i)
        
        return parameters
    

    def get_operational_factor(self, operational_df, year):
        """
        Function returns the lmh
        CO2e/unit factor from imported operational
        emissions data for easy input into functions.
        operational_df = import_emission_data() output [1]
        year = year int
        """
        o_dict = dict(zip(['Single Family','Missing Middle','Mid High Rise'], 
                            operational_df.loc[year,['detached_gju','attached_gju','apartment_gju']].values*10**3))
        
        return o_dict
    

    #####################################
    # Simulation functions
    #####################################

    ## Building block
    def build_embodied(self, n, e_params, distribution=stats.lognorm):
        """
        Building block function that calculates embodied
        emissions from n constructed houses in time period T.
        n = # units to build
        e_dist = distribution of embodied emissions (kgCO2e/unit)
        e_params = embodied distribution parameters
        """
        # sample embodied
        e_samples = distribution.rvs(*e_params, size=n)
        e_emissions = np.sum(e_samples) # sum of CO2e

        return e_emissions


    def build_operational(self, n, o_factor):
        """
        Building block function that calculates operational (energy)
        emissions from n constructed houses in time period T
        n = # of units
        o_factor = operational CO2e per unit in T
        """
        # return performance variables
        o_emissions = o_factor * n # CO2e/unit * unit/time

        return o_emissions
    

    ## (l,m,h) functions
    def split_n_lmh(self, n, l=0.09, m=0.24, h=0.67):
        """
        splits # built n into the three proportions of
        l (single family), m (low-rise multi-unit), 
        h (mid high rise). Rounding up.
        """
        # get n for each type
        n_l = int(np.ceil(n*l))
        n_m = int(np.ceil(n*m))
        n_h = int(np.ceil(n*h))
        n_dict = dict(zip(self.lmh,[n_l,n_m,n_h]))

        return n_dict
    
    
    def build_embodied_lmh(self, n, e_params_dict, distribution=stats.lognorm, l=0.09, m=0.24, h=0.67):
        """
        Extend build_embodied to allow for control of
        percentage of housing types built 
        (single family, low-rise multi-unit, mid/high rise)

        dict form -> {'Single Family','Missing Middle','Mid High Rise'}
        e_params_dict = full dict from embodied_params_exp_norm
        """
        # split n
        n_dict = self.split_n_lmh(n, l, m, h)

        lmh_embodied = {}
        for h_type in self.lmh:
            b = self.build_embodied(n=n_dict[h_type], 
                                    e_params=e_params_dict[h_type],
                                    distribution=distribution)
            # add n_dict to the output
            #lmh_embodied['built'] = n_dict[h_type]
            lmh_embodied[h_type] = b

        return lmh_embodied
    

    def build_operational_lmh(self, n, o_factor_dict, l=0.09, m=0.24, h=0.67):
        """
        Extend build_operational to allow for control of
        percentage of housing types built 
        (single family, low-rise multi-unit, mid/high rise)

        dict form -> {'Single Family','Missing Middle','Mid High Rise'}
        o_factor_dict = dictionary of operational factors
        """
        # split n
        n_dict = self.split_n_lmh(n, l, m, h)

        lmh_operational = {}
        for h_type in self.lmh:
            b = self.build_operational(n=n_dict[h_type],
                                       o_factor=o_factor_dict[h_type])
            # add n_dict to the output
            lmh_operational[h_type] = b

        return lmh_operational


    #####################################
    # Full simulation - uses object variables.
    #####################################   
    def chain_periods_build_lmh(self, y, b, verbose=False):
        """
        chain together multiple time periods of the simulation
        (e.g. years), store cumulative emissions, calculate 
        the continued operational output with the delta reduction
        each year.
        y = vector of time periods (int list).
        b = vector of # of houses built in time period (int list).
        e_data = embodied emissions data (labelled by year)
        o_data = operational emissions data (labelled by year)
        crn = common random numbers (default false)
        verbose = extra text outputs
        """
        # setup for storing variables
        raw = []
        embodied = []
        operational = [] # energy from houses built in current AND previous years.
        built = []
        
        # counting built
        nl_cum = 0
        nm_cum = 0
        nh_cum = 0

        # for each year
        for en, i in enumerate(y):
            # ---FIT INPUT DATA---
            # MLE for embodied, factor for operational
            embodied_params_i = self.embodied_fitter_mle(data=self.e_data.loc[self.e_data['year']==i], 
                                                         distribution=self.e_distribution)
            operational_factor_i = self.get_operational_factor(operational_df=self.o_data, 
                                                             year=i)

            # get the number of starts for each housing type
            n_lmh_i = self.split_n_lmh(n=b[en], 
                                       l=self.nlmh[0], m=self.nlmh[1], h=self.nlmh[2])


            # ---SIMULATE: BUILD HOUSES, CALC PERFORMANCE VARIABLES---
            # embodied emissions for the given year
            embodied_i = self.build_embodied_lmh(n=b[en], e_params_dict=embodied_params_i,
                                                 distribution=self.e_distribution,
                                                 l=self.nlmh[0], m=self.nlmh[1], h=self.nlmh[2])

            # operational emissions of cumulative houses built this year. ##########
            operational_i = self.build_operational_lmh(n=b[en], o_factor_dict=operational_factor_i,
                                                       l=self.nlmh[0], m=self.nlmh[1], h=self.nlmh[2])
            # energy emissions from houses built in previous years.
            operational_cum_l = self.build_operational(nl_cum, operational_factor_i['Single Family'])
            operational_cum_m = self.build_operational(nm_cum, operational_factor_i['Missing Middle'])
            operational_cum_h = self.build_operational(nh_cum, operational_factor_i['Mid High Rise'])

            if verbose == True:
                print('year', i)
                print('nlcum', nl_cum)
                print('opi', operational_i)
                print('opm', operational_cum_m)

            
            # ---STORE OUTPUTS---
            # full data
            raw.append({'year':i,
                        'embodied':embodied_i, 
                        'operational_y':operational_i,
                        'operational_cum':operational_i['Single Family']+operational_i['Missing Middle']+operational_i['Mid High Rise']+
                            operational_cum_l+operational_cum_m+operational_cum_h,
                        'num_built':n_lmh_i})

            # emissions each year
            embodied.append(embodied_i['Single Family']+embodied_i['Missing Middle']+embodied_i['Mid High Rise'])
            operational.append(operational_i['Single Family']+operational_i['Missing Middle']+operational_i['Mid High Rise']+
                            operational_cum_l+operational_cum_m+operational_cum_h)

            # number built
            built.append(n_lmh_i['Single Family'] + n_lmh_i['Missing Middle'] + n_lmh_i['Mid High Rise'])
            # broken down by type for energy calculation
            nl_cum += n_lmh_i['Single Family']
            nm_cum += n_lmh_i['Missing Middle']
            nh_cum += n_lmh_i['Mid High Rise']
    
        return {'raw':raw, 'E_e':embodied, 'E_o':operational, 'B':built}
    


    #####################################
    # Single year simulation for optimization - uses object variables.
    #####################################  
    # single year using the simulation object
    def single_year_simulation(self, b, b_cumulative):
        """
        Will calculate the emissions for self.year
        including cumulative operational emissions.
        b = houses built in year y
        b_cumulative = houses built in years 1->y
        """
        # ---FIT INPUT DATA---
        # MLE for embodied, factor for operational
        embodied_params_i = self.embodied_fitter_mle(data=self.e_data.loc[self.e_data['year']==self.year], 
                                                        distribution=self.e_distribution)
        operational_factor_i = self.get_operational_factor(operational_df=self.o_data, 
                                                            year=self.year)

        # get the number of starts for each housing type in 2030
        n_lmh_2030 = self.split_n_lmh(n=b, l=self.nlmh[0], m=self.nlmh[1], h=self.nlmh[2])
        # get the cumulative starts to 2030
        n_lmh_2030_cumulative = self.split_n_lmh(n=b_cumulative, l=self.nlmh[0], m=self.nlmh[1], h=self.nlmh[2])


        # ---SIMULATE: BUILD HOUSES, CALC PERFORMANCE VARIABLES---
        # embodied emissions for the given year
        embodied_i = self.build_embodied_lmh(n=b, e_params_dict=embodied_params_i,
                                                distribution=self.e_distribution,
                                                l=self.nlmh[0], m=self.nlmh[1], h=self.nlmh[2])

        # operational emissions of cumulative houses built
        operational_i = self.build_operational_lmh(n=b_cumulative, o_factor_dict=operational_factor_i,
                                                    l=self.nlmh[0], m=self.nlmh[1], h=self.nlmh[2])

        # ---STORE OUTPUTS---
        # full data
        raw = {'year':self.year,
                    'embodied':embodied_i, 
                    'operational_y':operational_i,
                    'num_built':n_lmh_2030_cumulative}

        # emissions each year
        embodied = embodied_i['Single Family']+embodied_i['Missing Middle']+embodied_i['Mid High Rise']
        operational = operational_i['Single Family']+operational_i['Missing Middle']+operational_i['Mid High Rise']

        # number built
        built = n_lmh_2030_cumulative['Single Family'] + n_lmh_2030_cumulative['Missing Middle'] + n_lmh_2030_cumulative['Mid High Rise']

        return {'raw':raw, 'E_e':embodied, 'E_o':operational, 'B':built}


            