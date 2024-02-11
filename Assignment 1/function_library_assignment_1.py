#NOTE: This library was written and used by Julia and Markus in course 46750 - we'll be using it here as well.
import numpy as np
import pandas as pd

#Assignment variables applicable across all tasks
n_bus = 24
wind_MWp = 300
S_base_3ph = 100

def read_data(data:str, path:str = 'data/'):
    #path is the path to the folder containing the files for the assignment

    #data specifies which data should be read and processed:
    # 'gen_costs'
    # 'gen_data'
    # 'line_data'
    # 'system_demand'
    # 'load_distribution'
    # 'wind_data_raw'
    # 'wind_data'
    # 'branch_matrix'

    gen_costs = pd.read_excel(path + 'gen_costs.xlsx')
    gen_costs['C (DKK/MWh)'] = gen_costs['C ($/MWh)'].values * 7.03 #USD to DKK

    gen_data = pd.read_excel(path + 'gen_data.xlsx')
    line_data = pd.read_excel(path + 'line_data.xlsx')

    system_demand = pd.read_excel(path + 'system_demand.xlsx')
    load_distribution = pd.read_excel(path + 'load_distribution.xlsx')

    all_bid_prices = pd.read_excel(path + 'demand_bids.xlsx')

    # Wind data processing - see the notebook for visualization
    wind_data_raw = []
    for i in range(6):
        wt = pd.read_csv(path + 'wind %d.out' % (i + 1))
        wt = wt.drop(columns = ['Unnamed: 0']) #dropping the unnecessary index column
        wt['Mean'] = wt.mean(numeric_only=True, axis=1)
        wind_data_raw.append(wt)

    wind_data = pd.DataFrame(index = wind_data_raw[0].index, data = wind_data_raw[0]['Mean'].values * wind_MWp, columns=['WF1 MW'])

    for i in range(1,6):
        wind_data['WF%d MW' % (i + 1)] = wind_data_raw[i]['Mean'].values * wind_MWp #from normalized to MW values

    wind_data = wind_data[12:36].reset_index(drop=True) #selected data is from hours 13 to and including 36 (0-indexed: 12 to and including 35)

    # It is proposed in the system report that the transmission lines connecting the 
    # node pairs (15, 21), (14, 16) and (13, 23) is reduced to 400 MW, 250 MW and 250 MW, respectively
    line_data.loc[(line_data['From'] == 15) & (line_data['To'] == 21), 'Capacity MVA'] = 400
    line_data.loc[(line_data['From'] == 14) & (line_data['To'] == 16), 'Capacity MVA'] = 250
    line_data.loc[(line_data['From'] == 13) & (line_data['To'] == 23), 'Capacity MVA'] = 250

    #Filling out branch susceptance matrix and adjusting branch data
    line_data['Susceptance pu'] = 1 / line_data['Reactance pu']
    line_data['Capacity pu'] = line_data['Capacity MVA'] / S_base_3ph
    
    branch_matrix = np.zeros((n_bus, n_bus))

    for n in range(1, n_bus + 1):
        branch_matrix[n - 1][n - 1] = line_data.loc[(line_data['From'] == n) | (line_data['To'] == n), 'Susceptance pu'].sum()
        for k in range(n, n_bus + 1): #start from n
            if k != n:
                branch_matrix[n - 1][k - 1] = -1 * (line_data.loc[(line_data['From'] == n) & (line_data['To'] == k), 'Susceptance pu'].sum())
                branch_matrix[k - 1][n - 1] = branch_matrix[n - 1][k - 1] #symmetric

    #Since the process is so fast, we might as well just handle all the data at every function call and return the desired data
    
    if data == 'gen_costs':
        return gen_costs
    elif data == 'gen_data':
        return gen_data
    elif data == 'line_data':
        return line_data
    elif data == 'system_demand':
        return system_demand
    elif data == 'load_distribution':
        return load_distribution
    elif data == 'wind_data_raw':
        return wind_data_raw
    elif data == 'wind_data':
        return wind_data
    elif data == 'branch_matrix':
        return branch_matrix
    elif data == 'all_bid_prices':
        return all_bid_prices  
    else:
        print('Invalid input.')
        return None

def mapping_dictionaries(gen_data, wf_nodes:list = [2, 4, 6, 15, 20, 22]):
    #These dictionarys return the 0-indexed indices of generators or wind farms located at the node n

    #for example, gens_map.get(0) will return the list [0] because generator 1 is placed at node 1
    #gens_map.get(14) will return [4, 5] because generators 5 and 6 are at node 15

    #if no wind farms or generators are located at node n, it will return an empty list

    gens_map = {}

    for n in range(1, n_bus + 1):
        gens_map[n - 1] = (gen_data['Unit #'][gen_data['Node'] == n] - 1).tolist()

    wf_map = {}

    for n in range(n_bus):
        wf_map[n] = np.where(np.array(wf_nodes) == n)[0].tolist()

    return gens_map, wf_map