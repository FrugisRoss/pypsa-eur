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
fn = "resources/Iberic100_2035_10ccslimit_noFR_24h/all/networks/base_s_40__24h_2035.nc"
n = pypsa.Network(fn)
config = yaml.safe_load(Path("config/config.iberic100_2035_modco2_24h_noFR.yaml").read_text())


nodes = n.buses.loc[
    n.buses.index.str[:2].isin(config['countries']) &
    (n.buses['carrier'] == 'AC')
].index.tolist()



#%%

def separate_co2_links(n):

    ''' Creates two series of links, one for DAC technolgies and one for industrial carbon pointsources,
        takes the network as an input '''
    co2_stored_producing_links = n.links.loc[(n.links['bus1'].str.contains('co2 stored') & (n.links['efficiency'] > 0)) |
                                             (n.links['bus2'].str.contains('co2 stored') & (n.links['efficiency2'] > 0))|
                                             (n.links['bus3'].str.contains('co2 stored') & (n.links['efficiency3'] > 0))|
                                             (n.links['bus4'].str.contains('co2 stored') & (n.links['efficiency4'] > 0))]

    dac_links=co2_stored_producing_links.loc[co2_stored_producing_links.index.str.contains('DAC')].index.to_series()
    industrial_links=co2_stored_producing_links.loc[~co2_stored_producing_links.index.str.contains('DAC')].index.to_series()

    return dac_links, industrial_links


dac_links, industrial_links = separate_co2_links(n)


def add_co2_stored_industrial_bus(n, nodes, industrial_links):

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
    

n=add_co2_stored_industrial_bus(n, nodes, industrial_links)

def add_co2_stored_dac_bus(n, nodes, dac_links):

    '''Renames the bus of the DAC links connected to co2 stored to 'co2 stored DAC' if the efficiency is positive,
        takes the network and the series of links as an input'''

    
    for node in nodes:

        n.add(
            "Bus",
            name=f"{node} co2 stored DAC",
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
        dac_links_at_node = dac_links[dac_links.str.startswith(node + ' ')]

        for link in dac_links_at_node:

            if ('co2 stored' in n.links.loc[link, 'bus1']) and (n.links.loc[link, 'efficiency'] >= 0):
                n.links.loc[link, 'bus1'] = f'{node} co2 stored DAC'
            elif ('co2 stored' in n.links.loc[link, 'bus2']) and (n.links.loc[link, 'efficiency2'] >= 0):
                n.links.loc[link, 'bus2'] = f'{node} co2 stored DAC'
            elif ('co2 stored' in n.links.loc[link, 'bus3']) and (n.links.loc[link, 'efficiency3'] >= 0):
                n.links.loc[link, 'bus3'] = f'{node} co2 stored DAC'
            elif ('co2 stored' in n.links.loc[link, 'bus4']) and (n.links.loc[link, 'efficiency4'] >= 0):
                n.links.loc[link, 'bus4'] = f'{node} co2 stored DAC'
    
    return n

n=add_co2_stored_dac_bus(n, nodes, dac_links)

def modify_co2_sequestered_link(n, nodes):

    '''Renames the bus of the co2 stored to 'co2 sequestered' if the efficiency is positive,
        takes the network and the series of links as an input'''

    
    for node in nodes:

        n.add(
            'Link',
            name=f"{node} co2 sequestered industrial",
            bus0=f"{node} co2 stored industrial",
            bus1=n.links.loc[f"{node} co2 sequestered", "bus1"],
            carrier=n.links.loc[f"{node} co2 sequestered", "carrier"],
            efficiency=n.links.loc[f"{node} co2 sequestered", "efficiency"],
            p_nom_extendable=n.links.loc[f"{node} co2 sequestered", "p_nom_extendable"],
            p_nom_min=n.links.loc[f"{node} co2 sequestered", "p_nom_min"],
            p_nom_max=n.links.loc[f"{node} co2 sequestered", "p_nom_max"],
            capital_cost=n.links.loc[f"{node} co2 sequestered", "capital_cost"],
            marginal_cost=n.links.loc[f"{node} co2 sequestered", "marginal_cost"],
            p_min_pu=n.links.loc[f"{node} co2 sequestered", "p_min_pu"],
            p_max_pu=n.links.loc[f"{node} co2 sequestered", "p_max_pu"],
            overwrite=True,
            reversed=False
        )

        n.add(
            'Link',
            name=f"{node} co2 sequestered DAC",
            bus0=f"{node} co2 stored DAC",
            bus1=n.links.loc[f"{node} co2 sequestered", "bus1"],
            carrier=n.links.loc[f"{node} co2 sequestered", "carrier"],
            efficiency=n.links.loc[f"{node} co2 sequestered", "efficiency"],
            p_nom_extendable=n.links.loc[f"{node} co2 sequestered", "p_nom_extendable"],
            p_nom_min=n.links.loc[f"{node} co2 sequestered", "p_nom_min"],
            p_nom_max=n.links.loc[f"{node} co2 sequestered", "p_nom_max"],
            capital_cost=n.links.loc[f"{node} co2 sequestered", "capital_cost"],
            marginal_cost=n.links.loc[f"{node} co2 sequestered", "marginal_cost"],
            p_min_pu=n.links.loc[f"{node} co2 sequestered", "p_min_pu"],
            p_max_pu=n.links.loc[f"{node} co2 sequestered", "p_max_pu"],
            overwrite=True,
            reversed=False
        )
        
        n.remove('Link',
                name=f"{node} co2 sequestered")

    
    return n

n=modify_co2_sequestered_link(n, nodes)


def add_co2_stored_links(n, nodes, ):

    '''Adds links from the industrial and DAC buses to the co2 sequestered bus,
        takes the network and the series of links as an input'''

    
    for node in nodes:

        n.add(
            'Link',
            name=f"{node} co2 stored industrial",
            bus0=f"{node} co2 stored industrial",
            bus1=f"{node} co2 stored",
            carrier='co2 stored',
            efficiency=1,
            p_nom_extendable=True,
            p_nom_min=0,
            p_nom_max=1e6,
            capital_cost=0,
            marginal_cost=0,
            p_min_pu=0,
            p_max_pu=1,
            overwrite=True,
            reversed=False
        )

        n.add(
            'Link',
            name=f"{node} co2 stored DAC",
            bus0=f"{node} co2 stored DAC",
            bus1=f"{node} co2 stored",
            carrier='co2 stored',
            efficiency=1,
            p_nom_extendable=True,
            p_nom_min=0,
            p_nom_max=1e6,
            capital_cost=0,
            marginal_cost=0,
            p_min_pu=0,
            p_max_pu=1,
            overwrite=True,
            reversed=False
        )
    
    return n


n=add_co2_stored_links(n, nodes)

#%%

n.export_to_netcdf(fn)

#%%