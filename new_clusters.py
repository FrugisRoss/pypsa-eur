#%%
import pypsa
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
from pypsa.plot import add_legend_lines, add_legend_patches, add_legend_semicircles
import yaml
from pathlib import Path
import pandas as pd

#%%
fn = "resources/Iberic100_2035_20ccslimit/all/networks/base_s_100__3h_2035.nc"
n = pypsa.Network(fn)
config = yaml.safe_load(Path("config/config.iberic100_2035_modco2_3h.yaml").read_text())

cluster_cost_reduction = 0
cluster_seq = True                              #only for pointsource cluster
cluster_size=50                               #only for renewables cluster
renewables={"solar",'solar-hsat','onwind'}      #only for renewables cluster


nodes = n.buses.loc[
    n.buses.index.str[:2].isin(config['countries']) &
    (n.buses['carrier'] == 'AC')
].index.tolist()

nodes_renewable_cluster = [node for node in nodes if f"{node} urban central heat" in n.buses.index]


#%%

def separate_co2_links(n):

    ''' Creates two series of links, one for DAC technolgies and one for industrial carbon pointsources,
        takes the network as an input '''
    co2_stored_connected_links = n.links.loc[n.links['bus1'].str.contains('co2 stored')|
                                                n.links['bus2'].str.contains('co2 stored')|
                                                n.links['bus3'].str.contains('co2 stored')|
                                                n.links['bus4'].str.contains('co2 stored')]

    dac_links=co2_stored_connected_links.loc[co2_stored_connected_links.index.str.contains('DAC')].index.to_series()
    industrial_links=co2_stored_connected_links.loc[~co2_stored_connected_links.index.str.contains('DAC')].index.to_series()

    return dac_links, industrial_links


dac_links, industrial_links = separate_co2_links(n)


def rename_co2_industrial_links(n, nodes, industrial_links):

    '''Renames the bus of the industrial links connected to co2 stored to 'co2 stored industrial' if the efficiency is positive,
        takes the network and the series of links as an input'''

    
    for node in nodes:

        n.add(
            "Bus",
            name=f"{node} co2 stored industrial",
            v_nom=n.buses.at[f"{node} co2 stored", "v_nom"],
            x=n.buses.at[f"{node} co2 stored", "x"],
            y=n.buses.at[f"{node} co2 stored", "y"],
            unit=n.buses.at[f"{node} co2 stored", "unit"],
            location=n.buses.at[f"{node} co2 stored", "location"],
            country=n.buses.at[f"{node} co2 stored", "country"],
            carrier=n.buses.at[f"{node} co2 stored", "carrier"],
            control=n.buses.at[f"{node} co2 stored", "control"],
            substation_lv=n.buses.at[f"{node} co2 stored", "substation_lv"],
            substation_off=n.buses.at[f"{node} co2 stored", "substation_off"],
        )
        industrial_links_at_node = industrial_links[industrial_links.str.startswith(node + ' ')]

        for link in industrial_links_at_node:

            if ('co2 stored' in n.links.loc[link, 'bus1']) and (n.links.loc[link, 'efficiency'] >= 0):
                n.links.loc[link, 'bus1'] = f'{node} co2 stored industrial'
            elif ('co2 stored' in n.links.loc[link, 'bus2']) and (n.links.loc[link, 'efficiency2'] >= 0):
                n.links.loc[link, 'bus2'] = f'{node} co2 stored industrial'
            elif ('co2 stored' in n.links.loc[link, 'bus3']) and (n.links.loc[link, 'efficiency3'] >= 0):
                n.links.loc[link, 'bus3'] = f'{node} co2 stored industrial'
            elif ('co2 stored' in n.links.loc[link, 'bus4']) and (n.links.loc[link, 'efficiency4'] >= 0):
                n.links.loc[link, 'bus4'] = f'{node} co2 stored industrial'
        
    return n
    

n=rename_co2_industrial_links(n, nodes, industrial_links)

#%%

#POINT SOURCE CLUSTER

def add_buses_of_pointsource_cluster(n, nodes):

    for node in nodes:
        if not n.buses.index.str.contains(f"{node} pointsource cluster").any():

            # electricity bus

            n.add(
                "Bus",
                name=f"{node} pointsource cluster",
                v_nom=n.buses.at[node, "v_nom"],
                x=n.buses.at[node, "x"],
                y=n.buses.at[node, "y"],
                unit=n.buses.at[node, "unit"],
                location=n.buses.at[node, "location"],
                country=n.buses.at[node, "country"],
                carrier=n.buses.at[node, "carrier"],
                control=n.buses.at[node, "control"],
                substation_lv=n.buses.at[node, "substation_lv"],
                substation_off=n.buses.at[node, "substation_off"],
            )

        if not n.buses.index.str.contains(f"{node} battery pointsource cluster").any():

            #battery bus

            n.add(
                "Bus",
                name=f"{node} battery pointsource cluster",
                v_nom=n.buses.at[f"{node} battery", "v_nom"],
                x=n.buses.at[f"{node} battery", "x"],
                y=n.buses.at[f"{node} battery", "y"],
                unit=n.buses.at[f"{node} battery", "unit"],
                location=n.buses.at[f"{node} battery", "location"],
                country=n.buses.at[f"{node} battery", "country"],
                carrier=n.buses.at[f"{node} battery", "carrier"],
                control=n.buses.at[f"{node} battery", "control"],
                substation_lv=n.buses.at[f"{node} battery", "substation_lv"],
                substation_off=n.buses.at[f"{node} battery", "substation_off"],
            )

        if not n.buses.index.str.contains(f"{node} H2 pointsource cluster").any():
            
            #H2 bus

            n.add(
                "Bus",
                name=f"{node} H2 pointsource cluster",
                v_nom=n.buses.at[f"{node} H2", "v_nom"],
                x=n.buses.at[f"{node} H2", "x"],
                y=n.buses.at[f"{node} H2", "y"],
                unit=n.buses.at[f"{node} H2", "unit"],
                location=n.buses.at[f"{node} H2", "location"],
                country=n.buses.at[f"{node} H2", "country"],
                carrier=n.buses.at[f"{node} H2", "carrier"],
                control=n.buses.at[f"{node} H2", "control"],
                substation_lv=n.buses.at[f"{node} H2", "substation_lv"],
                substation_off=n.buses.at[f"{node} H2", "substation_off"],
            )

        if not n.buses.index.str.contains(f"{node} methanol pointsource cluster").any():
            n.add(
                "Bus",
                name=f"{node} methanol pointsource cluster",
                v_nom=n.buses.at["EU methanol", "v_nom"],
                x=n.buses.at["EU methanol", "x"],
                y=n.buses.at["EU methanol", "y"],
                unit=n.buses.at["EU methanol", "unit"],
                location=n.buses.at["EU methanol", "location"],
                country=n.buses.at["EU methanol", "country"],
                carrier=n.buses.at["EU methanol", "carrier"],
                control=n.buses.at["EU methanol", "control"],
                substation_lv=n.buses.at["EU methanol", "substation_lv"],
                substation_off=n.buses.at["EU methanol", "substation_off"],
            )

    return n



def add_links_of_pointsource_cluster(n, nodes, cluster_cost_reduction, cluster_seq):

    for node in nodes:

        ### H2 Electrolysis ###

        link_name = f"{node} H2 Electrolysis"

        n.add(
            "Link",
            name=link_name + " pointsource cluster",
            bus0=n.links.at[link_name, "bus0"] + " pointsource cluster",
            bus1=n.links.at[link_name, "bus1"] + " pointsource cluster",
            p_nom_extendable=n.links.at[link_name, "p_nom_extendable"],
            carrier=n.links.at[link_name, "carrier"],
            efficiency=n.links.at[link_name, "efficiency"],
            capital_cost=n.links.at[link_name, "capital_cost"]*(1-cluster_cost_reduction),
            marginal_cost=n.links.at[link_name, "marginal_cost"]*(1-cluster_cost_reduction),
            lifetime=n.links.at[link_name, "lifetime"],
            reversed=False,
            overwrite=True,
        )

        ### Methanolization ###

        link_name = f"{node} methanolisation"
        cluster_methanol_bus_name = f"{node} methanol pointsource cluster"

        n.add(
            "Link",
            name=link_name + " pointsource cluster",
            bus0=n.links.at[link_name, "bus0"] + " pointsource cluster",
            bus1=cluster_methanol_bus_name,
            bus2=n.links.at[link_name, "bus2"] + " pointsource cluster",
            bus3=f'{node} co2 stored industrial',
            bus4=n.links.at[link_name, "bus4"],
            p_nom_extendable=n.links.at[link_name, "p_nom_extendable"],
            p_min_pu=n.links.at[link_name, "p_min_pu"],
            carrier=n.links.at[link_name, "carrier"],
            efficiency=n.links.at[link_name, "efficiency"],
            efficiency2=n.links.at[link_name, "efficiency2"],
            efficiency3=n.links.at[link_name, "efficiency3"],
            efficiency4=n.links.at[link_name, "efficiency4"],
            capital_cost=n.links.at[link_name, "capital_cost"]*(1-cluster_cost_reduction),
            marginal_cost=n.links.at[link_name, "marginal_cost"]*(1-cluster_cost_reduction),
            lifetime=n.links.at[link_name, "lifetime"],
            reversed=False,
            overwrite=True,
        )

        

        n.add(
            "Link",
            name=f"{node} methanol pointsource cluster",
            bus0=cluster_methanol_bus_name,
            bus1=n.links.at[link_name, "bus1"],
            p_nom_extendable=True,
            carrier=n.buses.loc[n.links.at[link_name, "bus1"], "carrier"],  
            efficiency=1.0,
            capital_cost=0.0,
            marginal_cost=0.0,
            reversed=False,
            overwrite=True,
        )

        ### Fischer-Tropsch ###

        link_name = f"{node} Fischer-Tropsch"

        n.add(
            "Link",
            name=link_name + " pointsource cluster",
            bus0=n.links.at[link_name, "bus0"] + " pointsource cluster",
            bus1=n.links.at[link_name, "bus1"],
            bus2=f'{node} co2 stored industrial',
            bus3=n.links.at[link_name, "bus3"],
            p_nom_extendable=n.links.at[link_name, "p_nom_extendable"],
            p_min_pu=n.links.at[link_name, "p_min_pu"],
            carrier=n.links.at[link_name, "carrier"],
            efficiency=n.links.at[link_name, "efficiency"],
            efficiency2=n.links.at[link_name, "efficiency2"],
            efficiency3=n.links.at[link_name, "efficiency3"],
            capital_cost=n.links.at[link_name, "capital_cost"]*(1-cluster_cost_reduction),
            marginal_cost=n.links.at[link_name, "marginal_cost"]*(1-cluster_cost_reduction),
            lifetime=n.links.at[link_name, "lifetime"],
            reversed=False,
            overwrite=True,
        )

        #maybe consider adding a bus specifically for the FT fuel produced in the cluster

        ### Sabatier ###

        link_name = f"{node} Sabatier"

        n.add(
            "Link",
            name=link_name + " pointsource cluster",
            bus0=n.links.at[link_name, "bus0"] + " pointsource cluster",
            bus1=n.links.at[link_name, "bus1"],
            bus2=f'{node} co2 stored industrial',
            bus3=n.links.at[link_name, "bus3"],
            p_nom_extendable=n.links.at[link_name, "p_nom_extendable"],
            p_min_pu=n.links.at[link_name, "p_min_pu"],
            carrier=n.links.at[link_name, "carrier"],
            efficiency=n.links.at[link_name, "efficiency"],
            efficiency2=n.links.at[link_name, "efficiency2"],
            efficiency3=n.links.at[link_name, "efficiency3"],
            capital_cost=n.links.at[link_name, "capital_cost"]*(1-cluster_cost_reduction),
            marginal_cost=n.links.at[link_name, "marginal_cost"]*(1-cluster_cost_reduction),
            lifetime=n.links.at[link_name, "lifetime"],
            reversed=False,
            overwrite=True,
        )

        #maybe consider adding a bus specifically for the Sabatier produced in the cluster

        if cluster_seq==True:

            ###Sequestration link

            link_name= f"{node} co2 sequestered pointsource cluster"

            n.add(
                "Link",
                name=link_name,
                bus0=f'{node} co2 stored industrial',
                bus1=f"{node} co2 sequestered",  #to try to connect it also to the co2 sequestered
                carrier=n.buses.at[f"{node} co2 sequestered", "carrier"],  
                p_nom_extendable=True,
                efficiency=1.0,
                capital_cost=0.000000000001,  #Costs to set
                marginal_cost=0.000000000001, #Costs to set
                reversed=False,
                overwrite=True,
            )
        

        ### Electricity connection to grid ###

        # link_name = f"{node} electricity pointsource cluster"
        
        # n.add(
        #     "Link",
        #     name=link_name,
        #     bus0=f"{node} pointsource cluster",
        #     bus1=f"{node}",
        #     carrier=n.buses.at[f"{node}", "carrier"],  
        #     p_nom_extendable=True,
        #     efficiency=1.0,
        #     capital_cost=0.0,
        #     marginal_cost=0.0,
        #     reversed=False,
        #     overwrite=True,
        # )
        # commented this out because it is assumed that the pointsource cluster can only import electricity

        link_name = f"{node} electricity pointsource cluster back"
        n.add(
            "Link",
            name=link_name,
            bus0=f"{node}",
            bus1=f"{node} pointsource cluster",
            carrier=n.buses.at[f"{node}", "carrier"],  
            p_nom_extendable=True,
            efficiency=1.0,
            capital_cost=0.0,
            marginal_cost=0.0,
            reversed=False,
            overwrite=True,
        )

    return n




def add_stores_of_pointsource_cluster(n, nodes, cluster_cost_reduction):

    for node in nodes:

            link_name = f"{node} H2 Store"

        
            n.add("Store",
                name=link_name + " pointsource cluster",
                bus=n.stores.at[link_name, "bus"] + " pointsource cluster",
                carrier=n.stores.at[link_name, "carrier"],
                e_nom_extendable=True,
                capital_cost=n.stores.at[link_name, "capital_cost"]*(1-cluster_cost_reduction),
                marginal_cost=n.stores.at[link_name, "marginal_cost"]*(1-cluster_cost_reduction),
                e_initial_per_period=n.stores.at[link_name, "e_initial_per_period"],
                e_cyclic=n.stores.at[link_name, "e_cyclic"],
                e_cyclic_per_period=n.stores.at[link_name, "e_cyclic_per_period"],
                overwrite=True,
                )
            
            link_name = f"{node} battery"


            n.add(
                    "Link",
                    name=link_name + " charger pointsource cluster",
                    bus0=f"{node} pointsource cluster",
                    bus1=f"{node} battery pointsource cluster",
                    carrier=n.buses.at[link_name, "carrier"],   
                    p_nom_extendable=True,
                    efficiency=1.0,
                    capital_cost=0.0,
                    marginal_cost=0.0,
                    reversed=False,
                    overwrite=True,
                )
            n.add(
                    "Link",
                    name=link_name + " discharger pointsource cluster",
                    bus0=f"{node} battery pointsource cluster",
                    bus1=f"{node} pointsource cluster",
                    carrier=n.buses.at[link_name, "carrier"],
                    p_nom_extendable=True,
                    efficiency=1.0,
                    capital_cost=0.0,
                    marginal_cost=0.0,
                    reversed=True,
                    overwrite=True,
                )

            n.add("Store",
                name=link_name + " pointsource cluster" ,
                bus=f"{node} battery pointsource cluster",
                carrier=n.stores.at[link_name, "carrier"],
                e_nom_extendable=True,
                capital_cost=n.stores.at[link_name, "capital_cost"]*(1-cluster_cost_reduction),
                marginal_cost=n.stores.at[link_name, "marginal_cost"]*(1-cluster_cost_reduction),
                e_initial_per_period=n.stores.at[link_name, "e_initial_per_period"],
                e_cyclic=n.stores.at[link_name, "e_cyclic"],
                e_cyclic_per_period=n.stores.at[link_name, "e_cyclic_per_period"],
                overwrite=True,
                )

            link_name = f"{node} co2 stored"

            n.add("Store",
                name=link_name + " pointsource cluster" ,
                bus=f'{node} co2 stored industrial',
                carrier=n.stores.at[link_name, "carrier"],
                e_nom_extendable=True,
                capital_cost=n.stores.at[link_name, "capital_cost"]*(1-cluster_cost_reduction),
                marginal_cost=n.stores.at[link_name, "marginal_cost"]*(1-cluster_cost_reduction),
                #capital_cost=0.00001,
                #marginal_cost=0.00001,
                e_initial_per_period=n.stores.at[link_name, "e_initial_per_period"],
                e_cyclic=n.stores.at[link_name, "e_cyclic"],
                e_cyclic_per_period=n.stores.at[link_name, "e_cyclic_per_period"],
                overwrite=True,
                )

    return n


def add_pointsource_cluster (n, nodes, cluster_cost_reduction, cluster_seq):

    n=add_buses_of_pointsource_cluster(n, nodes)
    n=add_links_of_pointsource_cluster(n, nodes, cluster_cost_reduction, cluster_seq)
    n=add_stores_of_pointsource_cluster(n, nodes, cluster_cost_reduction)

    return n    

# %%
#RENEWABLE CLUSTER

def add_buses_of_renewable_cluster(n, nodes):

    for node in nodes:
        if not n.buses.index.str.contains(f"{node} renewable cluster").any():

            # electricity bus

            n.add(
                "Bus",
                name=f"{node} renewable cluster",
                v_nom=n.buses.at[node, "v_nom"],
                x=n.buses.at[node, "x"],
                y=n.buses.at[node, "y"],
                unit=n.buses.at[node, "unit"],
                location=n.buses.at[node, "location"],
                country=n.buses.at[node, "country"],
                carrier=n.buses.at[node, "carrier"],
                control=n.buses.at[node, "control"],
                substation_lv=n.buses.at[node, "substation_lv"],
                substation_off=n.buses.at[node, "substation_off"],
            )

        if not n.buses.index.str.contains(f"{node} battery renewable cluster").any():

            #battery bus

            n.add(
                "Bus",
                name=f"{node} battery renewable cluster",
                v_nom=n.buses.at[f"{node} battery", "v_nom"],
                x=n.buses.at[f"{node} battery", "x"],
                y=n.buses.at[f"{node} battery", "y"],
                unit=n.buses.at[f"{node} battery", "unit"],
                location=n.buses.at[f"{node} battery", "location"],
                country=n.buses.at[f"{node} battery", "country"],
                carrier=n.buses.at[f"{node} battery", "carrier"],
                control=n.buses.at[f"{node} battery", "control"],
                substation_lv=n.buses.at[f"{node} battery", "substation_lv"],
                substation_off=n.buses.at[f"{node} battery", "substation_off"],
            )

        if not n.buses.index.str.contains(f"{node} H2 renewable cluster").any():
            
            #H2 bus

            n.add(
                "Bus",
                name=f"{node} H2 renewable cluster",
                v_nom=n.buses.at[f"{node} H2", "v_nom"],
                x=n.buses.at[f"{node} H2", "x"],
                y=n.buses.at[f"{node} H2", "y"],
                unit=n.buses.at[f"{node} H2", "unit"],
                location=n.buses.at[f"{node} H2", "location"],
                country=n.buses.at[f"{node} H2", "country"],
                carrier=n.buses.at[f"{node} H2", "carrier"],
                control=n.buses.at[f"{node} H2", "control"],
                substation_lv=n.buses.at[f"{node} H2", "substation_lv"],
                substation_off=n.buses.at[f"{node} H2", "substation_off"],
            )

        if not n.buses.index.str.contains(f"{node} methanol renewable cluster").any():

            #methanol bus

            n.add(
                "Bus",
                name=f"{node} methanol renewable cluster",
                v_nom=n.buses.at["EU methanol", "v_nom"],
                x=n.buses.at["EU methanol", "x"],
                y=n.buses.at["EU methanol", "y"],
                unit=n.buses.at["EU methanol", "unit"],
                location=n.buses.at["EU methanol", "location"],
                country=n.buses.at["EU methanol", "country"],
                carrier=n.buses.at["EU methanol", "carrier"],
                control=n.buses.at["EU methanol", "control"],
                substation_lv=n.buses.at["EU methanol", "substation_lv"],
                substation_off=n.buses.at["EU methanol", "substation_off"],
            )
        
        if not n.buses.index.str.contains(f"{node} co2 stored renewable cluster").any():

            #co2 bus
            
            n.add(
                "Bus",
                name=f"{node} co2 stored renewable cluster",
                v_nom=n.buses.at[f"{node} co2 stored", "v_nom"],
                x=n.buses.at[f"{node} co2 stored", "x"],
                y=n.buses.at[f"{node} co2 stored", "y"],
                unit=n.buses.at[f"{node} co2 stored", "unit"],
                location=n.buses.at[f"{node} co2 stored", "location"],
                country=n.buses.at[f"{node} co2 stored", "country"],
                carrier=n.buses.at[f"{node} co2 stored", "carrier"],
                control=n.buses.at[f"{node} co2 stored", "control"],
                substation_lv=n.buses.at[f"{node} co2 stored", "substation_lv"],
                substation_off=n.buses.at[f"{node} co2 stored", "substation_off"],
            )

    return n


def add_links_of_renewable_cluster(n, nodes, cluster_cost_reduction):

    for node in nodes:

        ### H2 Electrolysis ###

        link_name = f"{node} H2 Electrolysis"

        n.add(
            "Link",
            name=link_name + " renewable cluster",
            bus0=n.links.at[link_name, "bus0"] + " renewable cluster",
            bus1=n.links.at[link_name, "bus1"] + " renewable cluster",
            p_nom_extendable=n.links.at[link_name, "p_nom_extendable"],
            carrier=n.links.at[link_name, "carrier"],
            efficiency=n.links.at[link_name, "efficiency"],
            capital_cost=n.links.at[link_name, "capital_cost"]*(1-cluster_cost_reduction),
            marginal_cost=n.links.at[link_name, "marginal_cost"]*(1-cluster_cost_reduction),
            lifetime=n.links.at[link_name, "lifetime"],
            reversed=False,
            overwrite=True,
        )

        ### Methanolization ###

        link_name = f"{node} methanolisation"
        cluster_methanol_bus_name = f"{node} methanol renewable cluster"

        n.add(
            "Link",
            name=link_name + " renewable cluster",
            bus0=n.links.at[link_name, "bus0"] + " renewable cluster",
            bus1=cluster_methanol_bus_name,
            bus2=n.links.at[link_name, "bus2"] + " renewable cluster",
            bus3=f"{node} co2 stored renewable cluster",
            bus4=n.links.at[link_name, "bus4"],
            p_nom_extendable=n.links.at[link_name, "p_nom_extendable"],
            p_min_pu=n.links.at[link_name, "p_min_pu"],
            carrier=n.links.at[link_name, "carrier"],
            efficiency=n.links.at[link_name, "efficiency"],
            efficiency2=n.links.at[link_name, "efficiency2"],
            efficiency3=n.links.at[link_name, "efficiency3"],
            efficiency4=n.links.at[link_name, "efficiency4"],
            capital_cost=n.links.at[link_name, "capital_cost"]*(1-cluster_cost_reduction),
            marginal_cost=n.links.at[link_name, "marginal_cost"]*(1-cluster_cost_reduction),
            lifetime=n.links.at[link_name, "lifetime"],
            reversed=False,
            overwrite=True,
        )

        n.add(
            "Link",
            name=f"{node} methanol renewable cluster",
            bus0=cluster_methanol_bus_name,
            bus1=n.links.at[link_name, "bus1"],
            p_nom_extendable=True,
            carrier=n.buses.loc[n.links.at[link_name, "bus1"], "carrier"],  
            efficiency=1.0,
            capital_cost=0.0,
            marginal_cost=0.0,
            reversed=False,
            overwrite=True,
        )

        ### Fischer-Tropsch ###

        link_name = f"{node} Fischer-Tropsch"

        n.add(
            "Link",
            name=link_name + " renewable cluster",
            bus0=n.links.at[link_name, "bus0"] + " renewable cluster",
            bus1=n.links.at[link_name, "bus1"],
            bus2=f'{node} co2 stored renewable cluster',
            bus3=n.links.at[link_name, "bus3"],
            p_nom_extendable=n.links.at[link_name, "p_nom_extendable"],
            p_min_pu=n.links.at[link_name, "p_min_pu"],
            carrier=n.links.at[link_name, "carrier"],
            efficiency=n.links.at[link_name, "efficiency"],
            efficiency2=n.links.at[link_name, "efficiency2"],
            efficiency3=n.links.at[link_name, "efficiency3"],
            capital_cost=n.links.at[link_name, "capital_cost"]*(1-cluster_cost_reduction),
            marginal_cost=n.links.at[link_name, "marginal_cost"]*(1-cluster_cost_reduction),
            lifetime=n.links.at[link_name, "lifetime"],
            reversed=False,
            overwrite=True,
        )

        #maybe consider adding a bus specifically for the FT fuel produced in the cluster

        ### Sabatier ###

        link_name = f"{node} Sabatier"

        n.add(
            "Link",
            name=link_name + " renewable cluster",
            bus0=n.links.at[link_name, "bus0"] + " renewable cluster",
            bus1=n.links.at[link_name, "bus1"],
            bus2=f'{node} co2 stored renewable cluster',
            bus3=n.links.at[link_name, "bus3"],
            p_nom_extendable=n.links.at[link_name, "p_nom_extendable"],
            p_min_pu=n.links.at[link_name, "p_min_pu"],
            carrier=n.links.at[link_name, "carrier"],
            efficiency=n.links.at[link_name, "efficiency"],
            efficiency2=n.links.at[link_name, "efficiency2"],
            efficiency3=n.links.at[link_name, "efficiency3"],
            capital_cost=n.links.at[link_name, "capital_cost"]*(1-cluster_cost_reduction),
            marginal_cost=n.links.at[link_name, "marginal_cost"]*(1-cluster_cost_reduction),
            lifetime=n.links.at[link_name, "lifetime"],
            reversed=False,
            overwrite=True,
        )

        #maybe consider adding a bus specifically for the Sabatier produced in the cluster

        ### DAC ###

        link_name = f"{node} urban central DAC" #I chose urban central because we also connect the methanol plant to the urban central DH. Can be modified.

        n.add(
            "Link",
            name=link_name + " renewable cluster",
            bus0=n.links.at[link_name, "bus0"] + " renewable cluster",
            bus1=n.links.at[link_name, "bus1"],
            bus2=n.links.at[link_name, "bus2"],
            bus3=f"{node} co2 stored renewable cluster",
            bus4=n.links.at[link_name, "bus4"],
            p_nom_extendable=n.links.at[link_name, "p_nom_extendable"],
            p_min_pu=n.links.at[link_name, "p_min_pu"],
            carrier=n.links.at[link_name, "carrier"],
            efficiency=n.links.at[link_name, "efficiency"],
            efficiency2=n.links.at[link_name, "efficiency2"],
            efficiency3=n.links.at[link_name, "efficiency3"],
            efficiency4=n.links.at[link_name, "efficiency4"],
            capital_cost=n.links.at[link_name, "capital_cost"]*(1-cluster_cost_reduction),
            marginal_cost=n.links.at[link_name, "marginal_cost"]*(1-cluster_cost_reduction),
            lifetime=n.links.at[link_name, "lifetime"],
            reversed=False,
            overwrite=True,
        )

    return n


def add_stores_of_renewable_cluster(n, nodes, cluster_cost_reduction):

    for node in nodes:

            link_name = f"{node} H2 Store"

        
            n.add("Store",
                name=link_name + " renewable cluster",
                bus=n.stores.at[link_name, "bus"] + " renewable cluster",
                carrier=n.stores.at[link_name, "carrier"],
                e_nom_extendable=True,
                capital_cost=n.stores.at[link_name, "capital_cost"]*(1-cluster_cost_reduction),
                marginal_cost=n.stores.at[link_name, "marginal_cost"]*(1-cluster_cost_reduction),
                e_initial_per_period=n.stores.at[link_name, "e_initial_per_period"],
                e_cyclic=n.stores.at[link_name, "e_cyclic"],
                e_cyclic_per_period=n.stores.at[link_name, "e_cyclic_per_period"],
                overwrite=True,
                )
            
            link_name = f"{node} battery"


            n.add(
                    "Link",
                    name=link_name + " charger renewable cluster",
                    bus0=f"{node} renewable cluster",
                    bus1=f"{node} battery renewable cluster",
                    carrier=n.buses.at[link_name, "carrier"],   
                    p_nom_extendable=True,
                    efficiency=1.0,
                    capital_cost=0.0,
                    marginal_cost=0.0,
                    reversed=False,
                    overwrite=True,
                )
            n.add(
                    "Link",
                    name=link_name + " discharger renewable cluster",
                    bus0=f"{node} battery renewable cluster",
                    bus1=f"{node} renewable cluster",
                    carrier=n.buses.at[link_name, "carrier"],
                    p_nom_extendable=True,
                    efficiency=1.0,
                    capital_cost=0.0,
                    marginal_cost=0.0,
                    reversed=True,
                    overwrite=True,
                )

            n.add("Store",
                name=link_name + " renewable cluster" ,
                bus=f"{node} battery renewable cluster",
                carrier=n.stores.at[link_name, "carrier"],
                e_nom_extendable=True,
                capital_cost=n.stores.at[link_name, "capital_cost"]*(1-cluster_cost_reduction),
                marginal_cost=n.stores.at[link_name, "marginal_cost"]*(1-cluster_cost_reduction),
                e_initial_per_period=n.stores.at[link_name, "e_initial_per_period"],
                e_cyclic=n.stores.at[link_name, "e_cyclic"],
                e_cyclic_per_period=n.stores.at[link_name, "e_cyclic_per_period"],
                overwrite=True,
                )

            link_name = f"{node} co2 stored"

            n.add("Store",
                name=link_name + " renewable cluster" ,
                bus=f'{node} co2 stored renewable cluster',
                carrier=n.stores.at[link_name, "carrier"],
                e_nom_extendable=True,
                capital_cost=n.stores.at[link_name, "capital_cost"]*(1-cluster_cost_reduction),
                marginal_cost=n.stores.at[link_name, "marginal_cost"]*(1-cluster_cost_reduction),
                #capital_cost=0.00001,
                #marginal_cost=0.00001,
                e_initial_per_period=n.stores.at[link_name, "e_initial_per_period"],
                e_cyclic=n.stores.at[link_name, "e_cyclic"],
                e_cyclic_per_period=n.stores.at[link_name, "e_cyclic_per_period"],
                overwrite=True,
                )

    return n


def add_generators_of_renewable_cluster(n, cluster_size, cluster_cost_reduction, renewables, nodes_with_clusters):
    
    nodes_renewables_cf = {}                #dictionary of dataframes by node and renewable type, sorting the generators by average capacity factor (descending order)
    clusters_generators={}                      #dictionary of dataframes by node and renewable type, containing the generators assigned to the cluster  

    for node in nodes_with_clusters.copy():

        insufficient_generators = False  #if one node doesn't have enough renewable capacity even for one of the renewable in renewables, then the cluster cannot be created and we skip to the next node. 

        for renewable in renewables:

            nodes_renewables_cf[(node, renewable)] = pd.DataFrame(
                index=n.generators['p_nom_max'].loc[n.generators.index.astype(str).str.contains(f"{node} .*{renewable}$")].index,
                columns=["p_max_pu","p_nom_max"]  
            )

            clusters_generators[(node, renewable)] = pd.DataFrame()

            #we are considering the highest mean p_min_pu to determine the best generators per renewable available


            nodes_renewables_cf[(node, renewable)] ["p_max_pu"] = n.generators_t['p_max_pu'].loc[:, n.generators_t['p_max_pu'].columns.astype(str).str.contains(f"{node} .*{renewable}$")].mean()
            nodes_renewables_cf[(node, renewable)] ["p_nom_max"] = n.generators['p_nom_max'].loc[n.generators.index.astype(str).str.contains(f"{node} .*{renewable}$")]
            nodes_renewables_cf[(node, renewable)] ["p_nom_min"] = n.generators['p_nom_min'].loc[n.generators.index.astype(str).str.contains(f"{node} .*{renewable}$")]


            nodes_renewables_cf[(node, renewable)] = nodes_renewables_cf[(node, renewable)].sort_values("p_max_pu", ascending=False)
            nodes_renewables_cf[(node, renewable)] ["p_nom_avail"] = nodes_renewables_cf[(node, renewable)] ["p_nom_max"] - nodes_renewables_cf[(node, renewable)] ["p_nom_min"] #we make available for the cluster only what is not installed yet

            #print(nodes_renewables_cf[(country, renewable)])

            number_gen=0


            while nodes_renewables_cf[(node, renewable)].iloc[0:number_gen+1]["p_nom_avail"].sum() <= cluster_size:

                if number_gen >= len(nodes_renewables_cf[(node, renewable)]):


                    print(f"Not enough {renewable} capacity abailable at node {node} to reach cluster_size.")

                    nodes_with_clusters.remove(node)

                    insufficient_generators = True
                    break

                number_gen += 1

            if insufficient_generators:
                break  # exits the for renewable loop → goes to next node
    

            clusters_generators[(node, renewable)]  = n.generators.loc[nodes_renewables_cf[(node, renewable)].index[0:number_gen+1]]
            remaining_avail_capacity = nodes_renewables_cf[(node, renewable)].iloc[0:number_gen+1]["p_nom_avail"].sum() - cluster_size

            p_nom_available_last_cluster_gen= cluster_size - clusters_generators[(node, renewable)].loc[clusters_generators[(node, renewable)].index[0:number_gen],"p_nom_max"].sum() + clusters_generators[(node, renewable)].loc[clusters_generators[(node, renewable)].index[0:number_gen],"p_nom_min"].sum()
            
            clusters_generators[(node, renewable)].loc[clusters_generators[(node, renewable)].index[number_gen], "p_nom_max"] =clusters_generators[(node, renewable)].loc[clusters_generators[(node, renewable)].index[number_gen], "p_nom_min"] + p_nom_available_last_cluster_gen

            for idx in clusters_generators[(node, renewable)].index:

                ### Electricity generators ###

                p_nom_avail= clusters_generators[(node, renewable)].loc[idx].p_nom_max - clusters_generators[(node, renewable)].loc[idx].p_nom_min

                n.add(
                    "Generator",
                    name=clusters_generators[(node, renewable)].loc[idx].name + " renewable cluster",
                    bus=clusters_generators[(node, renewable)].loc[idx].bus + " renewable cluster",
                    carrier=clusters_generators[(node, renewable)].loc[idx].carrier,
                    p_nom_max=p_nom_avail,
                    p_max_pu=clusters_generators[(node, renewable)].loc[idx].p_max_pu,
                    marginal_cost=clusters_generators[(node, renewable)].loc[idx].marginal_cost*(1-cluster_cost_reduction),
                    capital_cost=clusters_generators[(node, renewable)].loc[idx].capital_cost*(1-cluster_cost_reduction),
                    efficiency=clusters_generators[(node, renewable)].loc[idx].efficiency,
                    p_nom_extendable=True,
                    overwrite=True,)


                
                
                n.generators_t['p_max_pu'][clusters_generators[(node, renewable)].loc[idx].name + " renewable cluster"] = n.generators_t['p_max_pu'][clusters_generators[(node, renewable)].loc[idx].name]

                # if idx == nodes_renewables_cf[(node, renewable)].iloc[number_gen].name:
                #     n.generators.loc[n.generators.index == idx, "p_nom_max"] =n.generators.loc[n.generators.index == idx, "p_nom_min"]+remaining_avail_capacity

                #     #print(f"Residual capacity of generator {clusters_generators[(node, renewable)].loc[idx].name} is {n.generators.loc[n.generators.index == idx, 'p_nom_max']} MW")
                
                # else:


                #     n.remove(
                #             "Generator",
                #             name=clusters_generators[(node, renewable)].loc[idx].name,
                #     )

    return n

#%%


# %%

def add_renewable_cluster(n, nodes, cluster_size, cluster_cost_reduction, renewables, nodes_with_clusters):

#all good here but we need to understand how to handle the case where there are not enough renewable generators available at a node to reach the cluster size.
#in that case we should remove  the 'insufficient' nodes from nodes_with_clusters
    n=add_buses_of_renewable_cluster(n, nodes)
    n=add_links_of_renewable_cluster(n, nodes, cluster_cost_reduction)
    n=add_stores_of_renewable_cluster(n, nodes, cluster_cost_reduction)
    n=add_generators_of_renewable_cluster(n, cluster_size, cluster_cost_reduction, renewables, nodes_with_clusters)

    return n


# %%

#add line to save the network

n=add_pointsource_cluster(n, nodes, cluster_cost_reduction, cluster_seq)

n=add_renewable_cluster(n, nodes_renewable_cluster, cluster_size, cluster_cost_reduction, renewables, nodes_renewable_cluster)



#%%
n.links.loc[n.links.index.str.contains('Sabatier')]

#%%
n.links.loc[n.links.index.str.contains('Fischer-Tropsch')]
#%%
n.links.loc[n.links.index.str.contains('methanolisation')]

#%%
n.export_to_netcdf(fn)
# %%
