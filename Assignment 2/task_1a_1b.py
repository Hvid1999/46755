#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gurobipy as gb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

plt.rcParams['font.size']=12
plt.rcParams['font.family']='serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False  
plt.rcParams['axes.spines.bottom'] = True     
plt.rcParams["axes.grid"] = True
plt.rcParams['grid.linestyle'] = '-.' 
plt.rcParams['grid.linewidth'] = 0.4


# # Read scenario data

# Refer to **read_data.ipynb** for insight regarding how the data is generated and structured.

# In[2]:


with open('Data/ALL_scenarios.json') as f:
    all_scenarios = json.load(f)

all_scenarios.keys()


# Set up constants

# In[3]:


OMEGA = 250 # number of scenarios to sample
PI = 1 / OMEGA # probability of each sampled scenario - assumed to be equal

S = len(all_scenarios.keys()) - 1 # number of total scenarios
T = 24 # number of hours

WIND_CAPACITY = 200 #MWh


# Randomly sample scenarios without replacement

# In[4]:


import random
random.seed(123)

# Sample scenarios without replacement
in_sample_scenarios = random.sample(range(S), 250)

print(in_sample_scenarios)


# Extract in-sample scenarios

# In[5]:


scenarios = {}

# Extract sampled scenarios from dictionary containing all scenarios
for i in range(len(in_sample_scenarios)):
    scenarios[str(i)] = all_scenarios[str(in_sample_scenarios[i])]
    scenarios[str(i)]['Original Index'] = in_sample_scenarios[i]
    
print('Number of extracted scenarios:', len(scenarios))


# # One-price Scheme

# *(Task 1.a)*

# ## Run model

# In[41]:


def solve_op_scheme(scenarios, WIND_CAPACITY, T, OMEGA):
    direction = gb.GRB.MAXIMIZE #Min / Max

    m = gb.Model() # Create a Gurobi model  
    m.setParam('OutputFlag', 0)

    #============= Variables =============
    p_DA = m.addVars(T, lb=0, ub=gb.GRB.INFINITY, name="p_DA") # day-ahead power bid
    delta = m.addVars(T, OMEGA, lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, name="delta") # decision variable for the power imbalance - can be negative
    price_coeff = m.addVars(T, OMEGA, lb=0, ub=gb.GRB.INFINITY, name="K") # price coefficient for the imbalance price wrt. the day-ahead price

    #============= Objective function =============
    # Set objective function - note that the day-ahead price is factored out of the sum
    obj = gb.quicksum(PI * scenarios[str(w)]['Spot Price [EUR/MWh]'][t] * (p_DA[t] + price_coeff[t,w] * delta[t,w]) for t in range(T) for w in range(OMEGA))
    m.setObjective(obj, direction)

    #============= Day-ahead power bid limits ============

    #Upper limit is the nominal wind power
    m.addConstrs(p_DA[t] <= WIND_CAPACITY for t in range(T))

    #============= Power imbalance definition (realized - bid) ===============
    m.addConstrs(delta[t,w] == scenarios[str(w)]['Wind Power [MW]'][t] - p_DA[t] for t in range(T) for w in range(OMEGA))

    #============= Price coefficient definition ===============
    # the system balance parameter is 0 if the system has a surplus and 1 if it has a deficit
    m.addConstrs(price_coeff[t,w] == 1.2 * scenarios[str(w)]['System Balance State'][t] + 0.9 * (1 - scenarios[str(w)]['System Balance State'][t]) for t in range(T) for w in range(OMEGA))

    #============= Display and run model =============
    m.update()
    #m.display()
    m.optimize()

    #============= Results =============
    results = {}
    if m.status == gb.GRB.OPTIMAL:
        #initialization
        for scenario in range(OMEGA):
            df = pd.DataFrame(columns=['Hour', 'DA Price [EUR/MWh]', 'Wind Power [MW]', 'DA Bid [MW]', 'Imbalance [MW]', 'DA Profit [EUR]', 'Balancing Profit [EUR]', 'System State', 'Balancing Price Coefficient'])
            
            for t in range(T):
                df.loc[t] = [t, 
                            scenarios[str(scenario)]['Spot Price [EUR/MWh]'][t], 
                            scenarios[str(scenario)]['Wind Power [MW]'][t], p_DA[t].x, 
                            delta[t,scenario].x, scenarios[str(scenario)]['Spot Price [EUR/MWh]'][t] * p_DA[t].x, 
                            price_coeff[t,scenario].x * scenarios[str(scenario)]['Spot Price [EUR/MWh]'][t] * delta[t,scenario].x, 
                            scenarios[str(scenario)]['System Balance State'][t], price_coeff[t,scenario].x]
            df['Total Profit [EUR]'] = df['DA Profit [EUR]'] + df['Balancing Profit [EUR]']

            df['Hour'] = df['Hour'].astype(int)
            df['System State'] = df['System State'].astype(int)
            df['System State'] = df['System State'].apply(lambda x: 'Deficit' if x == 1 else 'Surplus')
            df.set_index('Hour', inplace=True)
            results[scenario] = df.copy(deep=True)

        print('-----------------------------------------------')
        print('Objective value (expected profit): %.2f EUR' % m.objVal)
        print('-----------------------------------------------')
        print('Day-ahead bids:')
        average_hourly_profit = np.mean([results[w]['Total Profit [EUR]'] for w in range(OMEGA)], axis=0)

        summary = pd.DataFrame(columns=['Hour', 'DA Bid [MW]', 'Average Profit [EUR]', 'Average Wind [MW]', 'Median Wind [MW]', 'Average System State', 'Average Price Coefficient'])
        for t in range(T):
            summary.loc[t] = [t, p_DA[t].x, np.mean(average_hourly_profit[t]), np.mean([results[w]['Wind Power [MW]'][t] for w in range(OMEGA)]), np.median([results[w]['Wind Power [MW]'][t] for w in range(OMEGA)]), np.mean([scenarios[str(w)]['System Balance State'][t] for w in range(OMEGA)]), np.mean([price_coeff[t,w].x for w in range(OMEGA)])]

        summary['Hour'] = summary['Hour'].astype(int)
        summary.set_index('Hour', inplace=True)

        results['Summary'] = summary.copy(deep=True)

        for t in range(T):
            print('Hour %d | Bid: %.2f MW | Average Profit: %.2f EUR' % (t, p_DA[t].x, average_hourly_profit[t]))

        print('Sum of average profits: %.2f EUR' % np.sum(average_hourly_profit))
        print('-----------------------------------------------')
        print('Runtime: %f ms' % (m.Runtime * 1e3))
        return p_DA, results
    else:
        print("Optimization was not successful.")
        return None, None    


# In[37]:


p_DA, results = solve_op_scheme(scenarios, WIND_CAPACITY, T, OMEGA)
results['Summary']


# In[39]:


# Print results for scenario 3
print(results[3])


# ## Visualize results

# Plot vs. average "system state" and median realized wind power

# In[40]:


#DA_bids = [p_DA[t].x for t in range(T)]

fig = plt.figure(figsize=(12, 6))
ax1 = fig.gca()

#plot day-ahead bid for each hour
results['Summary']['DA Bid [MW]'].to_frame().plot.bar(ax = ax1, label='Day-ahead bid', color='green', edgecolor='black', linewidth=0.7, align='center', width=0.5)
ax1.set_xlabel('Hour')
ax1.set_ylabel('Power [MW]')


#plot average "system state" for each hour
average_system_state = np.mean([scenarios[str(w)]['System Balance State'] for w in range(OMEGA)], axis=0)

#create twinx
ax2 = ax1.twinx()

ax2.plot(average_system_state, color='black', linewidth=1, label='System State', marker='o', markersize=5, linestyle=':')

#plot median realized wind power across scenarios as black line
ax1.plot(results['Summary']['Median Wind [MW]'], color='maroon', linewidth=1, marker='o', markersize=5,linestyle='--', label='Median Realized Wind')

ax1.set_ylim([0, 1.05 * WIND_CAPACITY])

# ask matplotlib for the plotted objects and their labels
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='lower left')

ax1.set_title('Day-ahead Bids vs. Average System States & Median Scenario Wind', weight='bold')
ax1.set_axisbelow(True)
ax2.grid(False)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

ax2.set_ylim([-0.05,1.05])
ax2.set_yticks([0,1])
ax2.set_yticklabels(['Surplus', 'Deficit'])

ax2.set_ylabel('Average System State Across Scenarios')

ax2.spines[['top','right']].set_visible(True)

fig.tight_layout()
#plt.savefig('Figures/One-price_BID_vs_SYSTEMSTATE_and_WIND.png', dpi=300, bbox_inches='tight')
plt.show()


# <span style="color: red;">**Notes:**</span>
# * Most places have one-price schemes now since it rewards those who help the system and penalizes those who don't, whereas the **two-price** scheme only punishes bad performance.
# * The results seen in this optimization make sense when we can reliably predict if the system is in excess or deficit, but in reality you would want to bid closer to your expected production to avoid being penalized in the balancing market. **Maybe the results will be different if we use more scenarios?**
# * Energinet generally does not have rules in place for how much imbalance you are allowed to be in - in other places, this can be punished to a larger degree.
# * Behaviour will be dependent on scenario data, but wind tends to be pretty random.
# * If we had a larger sample of "System State Scenarios" it might tend to be more balanced.

# System State:
# * 0 = Excess
# * 1 = Deficit

# When there is often a deficit (higher balancing price), it is seen to save the capacity until the balancing market.

# # Two-price Scheme

# *(Task 1.b)*

# ## Run model

# In[42]:


OMEGA = len(scenarios.keys()) - 1 # number of scenarios
T = 24 # number of hours
PI = 1 / OMEGA # probability of each scenario - assumed to be equal
WIND_CAPACITY = 200 #MWh

def solve_tp_scheme(scenarios, WIND_CAPACITY, T, OMEGA):
    direction = gb.GRB.MAXIMIZE #Min / Max

    m = gb.Model() # Create a Gurobi model  

    m.setParam('OutputFlag', 0)

    #============= Variables =============
    p_DA = m.addVars(T, lb=0, ub=gb.GRB.INFINITY, name="p_DA") # day-ahead power bid

    delta = m.addVars(T, OMEGA, lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, name="delta") # decision variable for the power imbalance - can be negative
    delta_up = m.addVars(T, OMEGA, lb=0, ub=gb.GRB.INFINITY, name="delta_up") # surplus
    delta_down = m.addVars(T, OMEGA, lb=0, ub=gb.GRB.INFINITY, name="delta_down") # deficit

    imbalance_revenue = m.addVars(T, OMEGA, lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, name="I") # imbalance revenue - can be negative

    # binary variables used to control the two-price logic
    y = m.addVars(T, OMEGA, vtype=gb.GRB.BINARY, name="y")
    z = m.addVars(4, T, OMEGA, vtype=gb.GRB.BINARY, name="z")

    #============= Objective function =============
    # Set objective function
    obj = gb.quicksum(PI * (scenarios[str(w)]['Spot Price [EUR/MWh]'][t] * p_DA[t] + imbalance_revenue[t,w]) for t in range(T) for w in range(OMEGA))
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

        print('-----------------------------------------------')
        print('Objective value (expected profit): %.2f EUR' % m.objVal)
        print('-----------------------------------------------')
        print('Day-ahead bids:')
        average_hourly_profit = np.mean([results[w]['Total Profit [EUR]'] for w in range(OMEGA)], axis=0)

        summary = pd.DataFrame(columns=['Hour', 'DA Bid [MW]', 'Average Profit [EUR]', 'Average Wind [MW]', 'Median Wind [MW]', 'Average System State'])
        for t in range(T):
            summary.loc[t] = [t, p_DA[t].x, np.mean(average_hourly_profit[t]), np.mean([results[w]['Wind Power [MW]'][t] for w in range(OMEGA)]), np.median([results[w]['Wind Power [MW]'][t] for w in range(OMEGA)]), np.mean([scenarios[str(w)]['System Balance State'][t] for w in range(OMEGA)])]

        summary['Hour'] = summary['Hour'].astype(int)
        summary.set_index('Hour', inplace=True)

        results['Summary'] = summary.copy(deep=True)

        for t in range(T):
            print('Hour %d | Bid: %.2f MW | Average Profit: %.2f EUR' % (t, p_DA[t].x, average_hourly_profit[t]))

        print('Sum of average profits: %.2f EUR' % np.sum(average_hourly_profit))
        print('-----------------------------------------------')
        print('Runtime: %f ms' % (m.Runtime * 1e3))
        for scenario in range(OMEGA):
            for t in range(T):
                if np.round(sum([z[i,t,scenario].x for i in range(4)]), 4) != 3:
                    print('WARNING: SCENARIO %d | HOUR %d | z:' % (scenario, t), z[0,t,scenario].x, z[1,t,scenario].x, z[2,t,scenario].x, z[3,t,scenario].x)
        return p_DA, results
    else:
        print("Optimization was not successful.") 
        return None, None   


# In[26]:


p_DA, results = solve_tp_scheme(scenarios, WIND_CAPACITY, T, OMEGA)
results['Summary']


# Check that the z-variables are (likely) working as intended...

# In[ ]:


# for scenario in range(OMEGA):
#     for t in range(T):
#         if np.round(sum([z[i,t,scenario].x for i in range(4)]), 4) != 3:
#             print('WARNING: SCENARIO %d | HOUR %d | z:' % (scenario, t), z[0,t,scenario].x, z[1,t,scenario].x, z[2,t,scenario].x, z[3,t,scenario].x)


# Check that results make sense for different scenarios

# In[32]:


results[3] #check results for a specific scenario


# ## Visualize results

# In[34]:


#DA_bids = [p_DA[t].x for t in range(T)]

fig = plt.figure(figsize=(12, 6))
ax1 = fig.gca()

#plot day-ahead bid for each hour
results['Summary']['DA Bid [MW]'].to_frame().plot.bar(ax = ax1, label='Day-ahead bid', color='green', edgecolor='black', linewidth=0.7, align='center', width=0.5)
ax1.set_xlabel('Hour')
ax1.set_ylabel('Power [MW]')


#plot average "system state" for each hour
average_system_state = np.mean([scenarios[str(w)]['System Balance State'] for w in range(OMEGA)], axis=0)

#create twinx
ax2 = ax1.twinx()

ax2.plot(average_system_state, color='black', linewidth=1, label='System State', marker='o', markersize=5, linestyle=':')

#plot median realized wind power across scenarios as black line
ax1.plot(results['Summary']['Median Wind [MW]'], color='maroon', linewidth=1, marker='o', markersize=5,linestyle='--', label='Median Realized Wind')

ax1.set_ylim([0,1.05 *WIND_CAPACITY])

# ask matplotlib for the plotted objects and their labels
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

ax1.set_title('Day-ahead Bids vs. Average System States & Median Scenario Wind', weight='bold')
ax1.set_axisbelow(True)
ax2.grid(False)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

ax2.set_ylim([-0.05,1.05])
ax2.set_yticks([0,1])
ax2.set_yticklabels(['Surplus', 'Deficit'])

ax2.set_ylabel('Average System State Across Scenarios')

ax2.spines[['top','right']].set_visible(True)

fig.tight_layout()

#plt.savefig('Figures/Two-price_BID_vs_SYSTEMSTATE_and_WIND.png', dpi=300, bbox_inches='tight')
plt.show()


# Similar plot with day-ahead prices instead of realized wind

# In[35]:


#DA_bids = [p_DA[t].x for t in range(T)]

fig = plt.figure(figsize=(12, 6))
ax1 = fig.gca()

#plot day-ahead bid for each hour
results['Summary']['DA Bid [MW]'].to_frame().plot.bar(ax = ax1, label='Day-ahead bid', color='green', edgecolor='black', linewidth=0.7, align='center', width=0.5)
ax1.set_xlabel('Hour')
ax1.set_ylabel('Power [MW]')


#plot average "system state" for each hour
average_system_state = np.mean([scenarios[str(w)]['System Balance State'] for w in range(OMEGA)], axis=0)

#create twinx
ax2 = ax1.twinx()

#Average spot price across scenarios
average_spot_price = np.mean([scenarios[str(w)]['Spot Price [EUR/MWh]'] for w in range(OMEGA)], axis=0)

ax2.plot(average_spot_price, color='saddlebrown', linewidth=1, label='Average Spot Price', marker='o', markersize=5, linestyle=':')

#plot median realized wind power across scenarios as black line
ax1.plot(results['Summary']['Median Wind [MW]'], color='maroon', linewidth=1, marker='o', markersize=5,linestyle='--', label='Median Realized Wind')

ax1.set_ylim([0,1.05 *WIND_CAPACITY])

# ask matplotlib for the plotted objects and their labels
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

ax1.set_title('Day-ahead Bids vs. Average Spot Price & Median Scenario Wind', weight='bold')
ax1.set_axisbelow(True)
ax2.grid(False)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

ax2.set_ylim([0,120])

ax2.set_ylabel('Average Spot Price Across Scenarios [EUR/MWh]')

ax2.spines[['top','right']].set_visible(True)

fig.tight_layout()

#plt.savefig('Figures/Two-price_BID_vs_SPOT_and_WIND.png', dpi=300, bbox_inches='tight')

plt.show()

