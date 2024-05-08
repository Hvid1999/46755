#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gurobipy as gb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import random

plt.rcParams['font.size']=12
plt.rcParams['font.family']='serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False  
plt.rcParams['axes.spines.bottom'] = True     
plt.rcParams["axes.grid"] = True
plt.rcParams['grid.linestyle'] = '-.' 
plt.rcParams['grid.linewidth'] = 0.4


# In[2]:


with open('Data/ALL_scenarios.json') as f:
    all_scenarios = json.load(f)

all_scenarios.keys()

OMEGA = 250 # number of scenarios to sample
PI = 1 / OMEGA # probability of each sampled scenario - assumed to be equal

S = len(all_scenarios.keys()) - 1 # number of total scenarios
T = 24 # number of hours

WIND_CAPACITY = 200 #MWh

random.seed(123)

# Sample scenarios without replacement
in_sample_scenarios = random.sample(range(S), OMEGA)

#print(in_sample_scenarios)

scenarios = {}

# Extract sampled scenarios from dictionary containing all scenarios
for i in range(len(in_sample_scenarios)):
    scenarios[str(i)] = all_scenarios[str(in_sample_scenarios[i])]
    scenarios[str(i)]['Original Index'] = in_sample_scenarios[i]
    
#print('Number of extracted scenarios:', len(scenarios))


# In[3]:


alpha = 0.9
beta_values = np.arange(0,1 + 0.1, 0.1) #np.linspace(0, 1, 10) 


# # One-price Scheme **(unfinished)**

# In[ ]:


def cvr_op_scheme(scenarios, WIND_CAPACITY, T, OMEGA, alpha, beta_values, minimize_printouts=False, mip_gap = 1e-4):
    results_per_beta = {}
    p_DA_values_per_beta = {}

    for beta in beta_values:

        print('===============================================')
        print('Solving for beta = %.2f...' % beta)

        direction = gb.GRB.MAXIMIZE #Min / Max

        m = gb.Model() # Create a Gurobi model  

        m.setParam('OutputFlag', 0)
        
        if not mip_gap == 1e-4: m.setParam('MIPGap', mip_gap) #gurobi default is 1e-4

        #============= Variables =============
        p_DA = m.addVars(T, lb=0, ub=gb.GRB.INFINITY, name="p_DA")  # day-ahead power bid
        delta = m.addVars(T, OMEGA, lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, name="delta")  # power imbalance
        price_coeff = m.addVars(T, OMEGA, lb=0, ub=gb.GRB.INFINITY, name="K") # price coefficient for the imbalance price wrt. the day-ahead price
        eta = m.addVars(OMEGA, lb=0, ub=gb.GRB.INFINITY, name="n")  # auxiliary variable for risk-averse term
        zeta = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, name="zeta")  # VaR variable for risk-averse term

        #============= Objective function =============
        # Define objective function
        expected_value = gb.quicksum((1/OMEGA) * scenarios[str(w)]['Spot Price [EUR/MWh]'][t] * (p_DA[t] + price_coeff[t,w] * delta[t,w])
                                    for t in range(T) for w in range(OMEGA))
        
        cvar = zeta - (1 / (1 - alpha)) * gb.quicksum((1/OMEGA) * eta[w] for w in range(OMEGA))

        obj = (1 - beta) * expected_value + beta * cvar

        m.setObjective(obj, direction)

        #============= Constraints =============
        # Day-ahead power bid limits
        #Upper limit is the nominal wind power
        m.addConstrs(p_DA[t] <= WIND_CAPACITY for t in range(T))

        #============= Power imbalance definition (realized - bid) ===============
        m.addConstrs(delta[t,w] == scenarios[str(w)]['Wind Power [MW]'][t] - p_DA[t] for t in range(T) for w in range(OMEGA))

        #============= Price coefficient definition ===============
        # the system balance parameter is 0 if the system has a surplus and 1 if it has a deficit
        m.addConstrs(price_coeff[t,w] == 1.2 * scenarios[str(w)]['System Balance State'][t] + 0.9 * (1 - scenarios[str(w)]['System Balance State'][t]) for t in range(T) for w in range(OMEGA))

        #============= Conditional value at risk (CVaR) constraints ===============
        m.addConstrs(-gb.quicksum(
            scenarios[str(w)]['Spot Price [EUR/MWh]'][t] * (p_DA[t] + price_coeff[t,w] * delta[t,w])
            for t in range(T)) + zeta - eta[w] <= 0 for w in range(OMEGA))
        
        m.addConstrs(eta[w] >= 0 for w in range(OMEGA))

        #============= Display and run model =============
        m.update()
        m.optimize()

        #============= Results =============
        results = {}
        if m.status == gb.GRB.OPTIMAL:
            # Initialization
            for scenario in range(OMEGA):
                df = pd.DataFrame(columns=['Hour', 'DA Price [EUR/MWh]', 'Wind Power [MW]', 'DA Bid [MW]',
                                        'Imbalance [MW]', 'DA Profit [EUR]', 'Balancing Profit [EUR]',
                                        'System State', 'Balancing Price Coefficient'])
                for t in range(T):
                    df.loc[t] = [t,
                                scenarios[str(scenario)]['Spot Price [EUR/MWh]'][t],
                                scenarios[str(scenario)]['Wind Power [MW]'][t],
                                p_DA[t].x,
                                delta[t, scenario].x,
                                scenarios[str(scenario)]['Spot Price [EUR/MWh]'][t] * p_DA[t].x,
                                (scenarios[str(scenario)]['Spot Price [EUR/MWh]'][t] *
                                price_coeff[t, scenario].x *
                                delta[t, scenario].x),
                                scenarios[str(scenario)]['System Balance State'][t],
                                price_coeff[t, scenario].x]
                df['Total Profit [EUR]'] = df['DA Profit [EUR]'] + df['Balancing Profit [EUR]']

                df['Hour'] = df['Hour'].astype(int)
                df['System State'] = df['System State'].astype(int)
                df['System State'] = df['System State'].apply(lambda x: 'Deficit' if x == 1 else 'Surplus')
                df.set_index('Hour', inplace=True)
                results[scenario] = df.copy(deep=True)

                eta_values = [eta[w].x for w in range(OMEGA)]
                #delta_values = [delta[t, w].x for t in range(T) for w in range(OMEGA)]
                #price_coeff_values = [price_coeff[t, w].x for t in range(T) for w in range(OMEGA)]

            #Get CVaR, VaR, and expected profit  
            cvar_value = cvar.getValue()
            results['Expected Profit'] = expected_value.getValue()
            results['CVaR'] = cvar_value
            results['VaR'] = zeta.x
            p_DA_values = [p_DA[t].x for t in range(T)]

            print('-----------------------------------------------')
            print('Objective value: %.2f EUR' % m.objVal)
            print('Expected Profit: %.2f EUR' % results['Expected Profit'])
            print('CVaR: %.10f EUR' % results['CVaR'])
            print('VaR: %.2f EUR' % results['VaR'])
            print('Eta values:', eta_values)
            #print('Delta values:', delta_values)
            #print('Price coefficient values:', price_coeff_values)
            
            if not minimize_printouts:
                print('-----------------------------------------------')
                print('Day-ahead bids:')

            average_hourly_profit = np.mean([results[w]['Total Profit [EUR]'] for w in range(OMEGA)], axis=0)

            summary = pd.DataFrame(columns=['Hour', 'DA Bid [MW]', 'Average Profit [EUR]', 'Average Wind [MW]',
                                            'Median Wind [MW]', 'Average System State', 'Average Price Coefficient'])
            for t in range(T):
                summary.loc[t] = [t, p_DA[t].x, np.mean(average_hourly_profit[t]),
                                np.mean([results[w]['Wind Power [MW]'][t] for w in range(OMEGA)]),
                                np.median([results[w]['Wind Power [MW]'][t] for w in range(OMEGA)]),
                                np.mean([scenarios[str(w)]['System Balance State'][t] for w in range(OMEGA)]),
                                np.mean([price_coeff[t, w].x for w in range(OMEGA)])]

            summary['Hour'] = summary['Hour'].astype(int)
            summary.set_index('Hour', inplace=True)

            results['Summary'] = summary.copy(deep=True)

            if not minimize_printouts:
                for t in range(T):
                    print('Hour %d | Bid: %.2f MW | Average Profit: %.2f EUR' % (t, p_DA[t].x, average_hourly_profit[t]))

                print('Sum of average profits: %.2f EUR' % np.sum(average_hourly_profit))
                print('-----------------------------------------------')
                print('Runtime: %f ms' % (m.Runtime * 1e3))

            # Save results for this beta
            p_DA_values_per_beta[beta] = p_DA_values
            results_per_beta[beta] = results

        else:
            print("Optimization was not successful.")
    return results_per_beta,p_DA_values_per_beta



# In[250]:

#results_per_beta_op, p_DA_values_per_beta_op = cvr_op_scheme(scenarios, WIND_CAPACITY, T, OMEGA, alpha, beta_values, minimize_printouts=True)

#results_per_beta_op[0.5]['Summary'].round(2)


# In[268]:


# Plot expected value vs. CVaR for different beta values

# fig = plt.figure(figsize=(10, 6))

# ax = fig.gca()
# betas = list(results_per_beta_op.keys())
# expected_profit = [results_per_beta_op[beta]['Expected Profit'] / 1e3 for beta in betas]
# cvar_values = [results_per_beta_op[beta]['CVaR'] / 1e3 for beta in betas]


# # Adjust bbox and arrowprops as needed
# arrowprops = dict(linestyle="dotted", arrowstyle='-', color='black', alpha=1)
# bbox_props = dict(boxstyle="round,pad=0.25", fc="lightgrey", ec="black", lw=1, linestyle=':', alpha=1)

# offsets = [(25,15),   # 0.0
#            (-10,15),  # 0.1
#            (-5,15),   # 0.2
#            (-25,-10), # 0.3
#            (-30,-25), # 0.4
#            (-5,-25),  # 0.5
#            (-25, 35), # 0.6
#            (5, 45),   # 0.7
#            (25,25),   # 0.8
#            (30,0),    # 0.9
#            (-75,35)]  # 1.0


# for i in range(len(betas)):
#     if i < (len(betas) - 1): 
#         label = '\u03B2 = %.1f' % betas[i]
#         xy = (cvar_values[i], expected_profit[i])
#     else: 
#         label = '\u03B2 = %.1f\n\n(Exp. profit = %.2f kEUR)' % (betas[i], expected_profit[i])
#         xy = (cvar_values[i], 140)

#     ax.annotate(label, 
#                     xy=xy, 
#                     xytext=offsets[i], 
#                     xycoords='data', 
#                     textcoords='offset points', 
#                     arrowprops=arrowprops,
#                     bbox=bbox_props,
#                     fontsize=8,
#                     color='maroon',
#                     ha = 'center',
#                     va = 'center',
#                     rotation=0,
#                     clip_on=False,
#                     annotation_clip=False)


# plt.plot(cvar_values, expected_profit, marker='o', linestyle='--', color='maroon', markersize=5, markeredgecolor='darkslategray', markeredgewidth=0.5, linewidth=1.2)
# plt.xlabel('CVaR [kEUR]')
# plt.ylabel('Expected Profit [kEUR]')

# #plt.yticks(np.arange(140, 161, 2.5))
# #plt.xticks(np.arange(0, 20, 2.5))

# plt.xlim([0,25000 / 1e3])
# ax.set_ylim([165000 / 1e3, 175000 / 1e3])

# plt.title('One-price: Expected Profit vs. CVaR for different values of \u03B2')

# #plt.savefig('Figures/One-price_Expected_Profit_vs_CVaR.png', dpi=300, bbox_inches='tight')

# plt.tight_layout()
# plt.show()


# In[266]:


# var_values = []
# betas = list(results_per_beta_op.keys())
# expected_profit = [results_per_beta_op[beta]['Expected Profit'] for beta in betas]
# var_values = [results_per_beta_op[beta]['VaR'] for beta in betas]

# # Plot VaR vs. beta
# plt.figure(figsize=(8, 6))
# plt.plot(beta_values[1:] , var_values[1:], marker='o', linestyle='-')
# plt.xlabel('Beta')
# plt.ylabel('VaR')
# plt.title('VaR vs. Beta')
# plt.grid(True)
# plt.ylim(28500, 29500)
# plt.margins(x=0)
# plt.show()


# # Two-price Scheme

# In[215]:


def cvr_tp_scheme(scenarios, WIND_CAPACITY, T, OMEGA, alpha, beta_values, minimize_printouts=False, mip_gap = 1e-4):

    results_per_beta = {}
    p_DA_values_per_beta = {}
    
    for beta in beta_values: 

        print('===============================================')
        print('Solving for beta = %.2f...' % beta)

        direction = gb.GRB.MAXIMIZE #Min / Max

        m = gb.Model() # Create a Gurobi model  

        m.setParam('OutputFlag', 0)
        
        if not mip_gap == 1e-4: m.setParam('MIPGap', mip_gap) #gurobi default is 1e-4

        #============= Variables =============
        p_DA = m.addVars(T, lb=0, ub=gb.GRB.INFINITY, name="p_DA") # day-ahead power bid
        delta = m.addVars(T, OMEGA, lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, name="delta") # decision variable for the power imbalance - can be negative
        delta_up = m.addVars(T, OMEGA, lb=0, ub=gb.GRB.INFINITY, name="delta_up") # surplus
        delta_down = m.addVars(T, OMEGA, lb=0, ub=gb.GRB.INFINITY, name="delta_down") # deficit
        eta = m.addVars(OMEGA, lb=0, ub=gb.GRB.INFINITY, name="n")  # auxiliary variable for risk-averse term
        zeta = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, name="zeta")  # VaR variable for risk-averse term

        imbalance_revenue = m.addVars(T, OMEGA, lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, name="I") # imbalance revenue - can be negative

        # binary variables used to control the two-price logic
        y = m.addVars(T, OMEGA, vtype=gb.GRB.BINARY, name="y")
        z = m.addVars(4, T, OMEGA, vtype=gb.GRB.BINARY, name="z")

        #============= Objective function =============
        # Set objective function
        expected_value = gb.quicksum((1/OMEGA) * (scenarios[str(w)]['Spot Price [EUR/MWh]'][t] * p_DA[t] + imbalance_revenue[t,w]) for t in range(T) for w in range(OMEGA))
        
        cvar = zeta - (1 / (1 - alpha)) * gb.quicksum((1/OMEGA) * eta[w] for w in range(OMEGA))
        obj = (1 - beta)*expected_value + beta*cvar

        m.setObjective(obj, direction)

        #============= Day-ahead power bid limits ============

        #Upper limit is the nominal wind power
        m.addConstrs(p_DA[t] <= WIND_CAPACITY for t in range(T))

        #============= Power imbalance definitions ===============
        m.addConstrs(delta[t,w] == scenarios[str(w)]['Wind Power [MW]'][t] - p_DA[t] for t in range(T) for w in range(OMEGA))
        m.addConstrs(delta[t,w] == delta_up[t,w] - delta_down[t,w] for t in range(T) for w in range(OMEGA))


        M = 1e6 # big-M constant
        #ensure that only one of the delta directions can be non-zero
        m.addConstrs(delta_up[t,w] <= M * (1 - y[t,w]) for t in range(T) for w in range(OMEGA))
        m.addConstrs(delta_down[t,w] <= M * y[t,w] for t in range(T) for w in range(OMEGA))

        #============= Linearized conditional statements ===============
        #Binary variable constraints
        m.addConstrs(z[0,t,w] <= y[t,w] + scenarios[str(w)]['System Balance State'][t] for t in range(T) for w in range(OMEGA))
        m.addConstrs(z[1,t,w] <= y[t,w] + (1 - scenarios[str(w)]['System Balance State'][t]) for t in range(T) for w in range(OMEGA))
        m.addConstrs(z[2,t,w] <= (1 - y[t,w]) + scenarios[str(w)]['System Balance State'][t] for t in range(T) for w in range(OMEGA))
        m.addConstrs(z[3,t,w] <= (1 - y[t,w]) + (1 - scenarios[str(w)]['System Balance State'][t]) for t in range(T) for w in range(OMEGA))

        # if system is in a surplus and the imbalance is positive (NOT helping the system)
        m.addConstrs(imbalance_revenue[t,w] <= 0.9 * scenarios[str(w)]['Spot Price [EUR/MWh]'][t] * delta_up[t,w] + M * z[0,t,w] for t in range(T) for w in range(OMEGA))

        # if system is in a deficit and the imbalance is positive (IS helping the system)
        m.addConstrs(imbalance_revenue[t,w] <= 1.0 * scenarios[str(w)]['Spot Price [EUR/MWh]'][t] * delta_up[t,w] + M * z[1,t,w] for t in range(T) for w in range(OMEGA))

        # if system is in a surplus and the imbalance is negative (IS helping the system)
        m.addConstrs(imbalance_revenue[t,w] <= -1.0 * scenarios[str(w)]['Spot Price [EUR/MWh]'][t] * delta_down[t,w] + M * z[2,t,w] for t in range(T) for w in range(OMEGA))

        # if system is in a deficit and the imbalance is negative (NOT helping the system)
        m.addConstrs(imbalance_revenue[t,w] <= -1.2 * scenarios[str(w)]['Spot Price [EUR/MWh]'][t] * delta_down[t,w] + M * z[3,t,w] for t in range(T) for w in range(OMEGA))

        #============= Conditional value at risk (CVaR) constraints ===============
        m.addConstrs(-gb.quicksum(
            scenarios[str(w)]['Spot Price [EUR/MWh]'][t] * p_DA[t] + imbalance_revenue[t,w]
            for t in range(T)) + zeta - eta[w] <= 0 for w in range(OMEGA))
        
        m.addConstrs(eta[w] >= 0 for w in range(OMEGA))

        #============= Display and run model =============
        m.update()
        #m.display()
        m.optimize()

        #============= Results =============
        results = {}
        if m.status == gb.GRB.OPTIMAL:
            #initialization
            for scenario in range(OMEGA):
                df = pd.DataFrame(columns=['Hour', 'DA Price [EUR/MWh]', 'Wind Power [MW]', 'DA Bid [MW]', 'Delta [MW]', 'Delta UP [MW]', 'Delta DOWN [MW]' ,'DA Profit [EUR]', 'Balancing Profit [EUR]', 'System State'])
                
                for t in range(T):
                    df.loc[t] = [t, 
                                scenarios[str(scenario)]['Spot Price [EUR/MWh]'][t], 
                                scenarios[str(scenario)]['Wind Power [MW]'][t], p_DA[t].x,
                                delta[t,scenario].x, 
                                delta_up[t,scenario].x, 
                                delta_down[t,scenario].x, 
                                scenarios[str(scenario)]['Spot Price [EUR/MWh]'][t] * p_DA[t].x, 
                                imbalance_revenue[t,scenario].x, 
                                scenarios[str(scenario)]['System Balance State'][t]]
                df['Total Profit [EUR]'] = df['DA Profit [EUR]'] + df['Balancing Profit [EUR]']

                df['Hour'] = df['Hour'].astype(int)
                df['System State'] = df['System State'].astype(int)
                df['System State'] = df['System State'].apply(lambda x: 'Deficit' if x == 1 else 'Surplus')
                df.set_index('Hour', inplace=True)
                results[scenario] = df.copy(deep=True)

                eta_values = [eta[w].x for w in range(OMEGA)]

            #Get CVaR, VaR, and expected profit    
            cvar_value = cvar.getValue()
            results['Expected Profit'] = expected_value.getValue()
            results['CVaR'] = cvar_value
            results['VaR'] = zeta.x
            p_DA_values = [p_DA[t].x for t in range(T)]

            print('-----------------------------------------------')
            print('Objective value: %.2f EUR' % m.objVal)
            print('Expected Profit: %.2f EUR' % results['Expected Profit'])
            print('CVaR: %.10f EUR' % results['CVaR'])
            print('VaR: %.2f EUR' % results['VaR'])
            print('Eta values:', eta_values)

            if not minimize_printouts:
                print('-----------------------------------------------')
                print('Day-ahead bids:')

            average_hourly_profit = np.mean([results[w]['Total Profit [EUR]'] for w in range(OMEGA)], axis=0)

            summary = pd.DataFrame(columns=['Hour', 'DA Bid [MW]', 'Average Profit [EUR]', 'Average Wind [MW]', 'Median Wind [MW]', 'Average System State'])
            for t in range(T):
                summary.loc[t] = [t, p_DA[t].x, np.mean(average_hourly_profit[t]), np.mean([results[w]['Wind Power [MW]'][t] for w in range(OMEGA)]), np.median([results[w]['Wind Power [MW]'][t] for w in range(OMEGA)]), np.mean([scenarios[str(w)]['System Balance State'][t] for w in range(OMEGA)])]

            summary['Hour'] = summary['Hour'].astype(int)
            summary.set_index('Hour', inplace=True)

            results['Summary'] = summary.copy(deep=True)

            if not minimize_printouts:
                for t in range(T):
                    print('Hour %d | Bid: %.2f MW | Average Profit: %.2f EUR' % (t, p_DA[t].x, average_hourly_profit[t]))

                print('Sum of average profits: %.2f EUR' % np.sum(average_hourly_profit))
                print('-----------------------------------------------')
                print('Runtime: %f ms' % (m.Runtime * 1e3))

            for scenario in range(OMEGA):
                for t in range(T):
                    if np.round(sum([z[i,t,scenario].x for i in range(4)]), 4) != 3:
                        print('WARNING: SCENARIO %d | HOUR %d | z:' % (scenario, t), z[0,t,scenario].x, z[1,t,scenario].x, z[2,t,scenario].x, z[3,t,scenario].x)
             # Save results for this beta
            p_DA_values_per_beta[beta] = p_DA_values
            results_per_beta[beta] = results

        else:
            print("Optimization was not successful.")
    return results_per_beta,p_DA_values_per_beta



# In[216]:
# results_per_beta_tp, p_DA_values_per_beta_tp = cvr_tp_scheme(scenarios, WIND_CAPACITY, T, OMEGA, alpha, beta_values, minimize_printouts=True)


# # Plot expected value vs. CVaR for different beta values

# fig = plt.figure(figsize=(10, 6))

# ax = fig.gca()
# betas = list(results_per_beta_tp.keys())
# expected_profit = [results_per_beta_tp[beta]['Expected Profit'] / 1e3 for beta in betas]
# cvar_values = [results_per_beta_tp[beta]['CVaR'] / 1e3 for beta in betas]


# # Adjust bbox and arrowprops as needed
# arrowprops = dict(linestyle="dotted", arrowstyle='-', color='black', alpha=1)
# bbox_props = dict(boxstyle="round,pad=0.25", fc="lightgrey", ec="black", lw=1, linestyle=':', alpha=1)

# offsets = [(25,15),   # 0.0
#            (-10,15),  # 0.1
#            (-5,15),   # 0.2
#            (-25,-10), # 0.3
#            (-30,-25), # 0.4
#            (-5,-25),  # 0.5
#            (-25, 35), # 0.6
#            (5, 45),   # 0.7
#            (25,25),   # 0.8
#            (30,0),    # 0.9
#            (-75,35)]  # 1.0


# for i in range(len(betas)):
#     if i < (len(betas) - 1): 
#         label = '\u03B2 = %.1f' % betas[i]
#         xy = (cvar_values[i], expected_profit[i])
#     else: 
#         label = '\u03B2 = %.1f\n\n(Exp. profit = %.2f kEUR)' % (betas[i], expected_profit[i])
#         xy = (cvar_values[i], 140)

#     ax.annotate(label, 
#                     xy=xy, 
#                     xytext=offsets[i], 
#                     xycoords='data', 
#                     textcoords='offset points', 
#                     arrowprops=arrowprops,
#                     bbox=bbox_props,
#                     fontsize=8,
#                     color='maroon',
#                     ha = 'center',
#                     va = 'center',
#                     rotation=0,
#                     clip_on=False,
#                     annotation_clip=False)


# plt.plot(cvar_values, expected_profit, marker='o', linestyle='--', color='maroon', markersize=5, markeredgecolor='darkslategray', markeredgewidth=0.5, linewidth=1.2)
# plt.xlabel('CVaR [kEUR]')
# plt.ylabel('Expected Profit [kEUR]')

# plt.yticks(np.arange(140, 161, 2.5))
# plt.xticks(np.arange(0, 20, 2.5))

# plt.xlim([0,19000 / 1e3])
# ax.set_ylim([140000 / 1e3, 160000 / 1e3])

# plt.title('Two-price: Expected Profit vs. CVaR for different values of \u03B2')

# #plt.savefig('Figures/Two-price_Expected_Profit_vs_CVaR.png', dpi=300, bbox_inches='tight')

# plt.tight_layout()
# plt.show()


# # Export optimal day-ahead schedule for subsequent tasks

# Create dataframe with optimal day-ahead bids for each approach - we pick the values for BETA = 0.5 as a "conservative" choice

# In[225]:


# BETA_EXPORT = 0.5 # the beta value to export day-ahead bids for

# df = pd.DataFrame(columns=['Hour', 'One-price Bids [MW]', 'Two-price Bids [MW]'])
# df['Hour'] = range(T)
# #df['One-price Bids [MW]'] = results_op['Summary']['DA Bid [MW]'].values
# df['Two-price Bids [MW]'] = results_per_beta_tp[BETA_EXPORT]['Summary']['DA Bid [MW]'].values
# df.set_index('Hour', inplace=True)

# #df.to_csv('Data/Optimal_DA_bids_from_CVaR.csv')

# df

