import pypsa
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
from pypsa.plot import add_legend_lines, add_legend_patches, add_legend_semicircles
import yaml
from pathlib import Path
import pandas as pd
from scripts._helpers import (
    configure_logging,
    get_snapshots,
    load_cutout,
    set_scenario_config,
)

# Pointsource clusters

def check_industrial_production_columns(industrial_production, industry_sector_ratios):
    """Check that the columns in industrial_production and industry_sector_ratios match."""

    extra_columns = set(industry_sector_ratios.columns) - set(industrial_production.columns)

    if extra_columns:
        raise ValueError(
            f"{', '.join(extra_columns)} found in industry_sector_ratios but not in industrial_production"
        )

    extra_columns = set(industrial_production.columns) - set(industry_sector_ratios.columns)  

    if extra_columns:
        raise ValueError(
            f"{', '.join(extra_columns)} found in industrial_production but not in industry_sector_ratios"
        )


    industry_sector_ratios = industry_sector_ratios[industrial_production.columns]
    return industrial_production, industry_sector_ratios

def find_nodes_eligible_for_carbon_clusters(n, industrial_production, industry_sector_ratios, co2_storage_treshold):    

    biomass_by_industry_by_node=pd.DataFrame()
    methane_by_industry_by_node=pd.DataFrame()
    process_emissions_by_industry_by_node=pd.DataFrame()
    carbon_available_by_industry_by_node=pd.DataFrame()
    nodes_with_carbon_clusters=[]


    for node in industrial_production.index:
        eff_biomass_CC = n.links.loc[
                n.links.index.str.contains(f'{node} solid biomass for industry CC'),
                'efficiency3'
            ].iloc[0]*n.links.loc[
                n.links.index.str.contains(f'{node} solid biomass for industry CC'),
                'efficiency'
            ].iloc[0]

        eff_methane_CC = n.links.loc[
                n.links.index.str.contains(f'{node} gas for industry CC'),
                'efficiency3'
            ].iloc[0]*n.links.loc[
                n.links.index.str.contains(f'{node} gas for industry CC'),
                'efficiency'
            ].iloc[0]

            
        for sector in industrial_production.columns:
            biomass_by_industry_by_node.loc[node,sector]=industrial_production.loc[node,sector]*industry_sector_ratios.loc['biomass',sector]*1e3 # MWh = ktonnes/year *MWh/tonne *tonnes/ktonnes
            methane_by_industry_by_node.loc[node,sector]=industrial_production.loc[node,sector]*industry_sector_ratios.loc['methane',sector]*1e3 # tonnesCO2 = ktonnes/year *MWh/tonne *tonnes/ktonnes
            process_emissions_by_industry_by_node.loc[node,sector]= industrial_production.loc[node,sector]*industry_sector_ratios.loc['process emission',sector]*1e3+industrial_production.loc[node,sector]*industry_sector_ratios.loc['process emission from feedstock',sector]*1e3 # tonnesCO2 = ktonnes/year * tonnesCO2/tonne * tonnes/ktonnes

            

            carbon_from_biomass=biomass_by_industry_by_node.loc[node,sector]*eff_biomass_CC
            carbon_from_methane=methane_by_industry_by_node.loc[node,sector]*eff_methane_CC


        

            carbon_available_by_industry_by_node.loc[node,sector]=carbon_from_biomass+carbon_from_methane+process_emissions_by_industry_by_node.loc[node,sector]

            if (carbon_available_by_industry_by_node.loc[node].sum()>co2_storage_treshold and node not in nodes_with_carbon_clusters):
                nodes_with_carbon_clusters.append(node)

    return nodes_with_carbon_clusters, carbon_available_by_industry_by_node, process_emissions_by_industry_by_node

def assign_co2_bus_to_carbon_clusters(n, nodes_with_carbon_clusters):

    for node in nodes_with_carbon_clusters:

        #define the CO2 bus for the carbon cluster
        bus = f'{node} co2 stored"

        n.add(
            "Bus",
            name=bus + " cluster",
            v_nom=n.buses.at[bus, "v_nom"],
            x=n.buses.at[bus, "x"],
            y=n.buses.at[bus, "y"],
            unit=n.buses.at[bus, "unit"],
            location=n.buses.at[bus, "location"],
            country=n.buses.at[bus, "country"],
            carrier=n.buses.at[bus, "carrier"],
            control=n.buses.at[bus, "control"],
            substation_lv=n.buses.at[bus, "substation_lv"],
            substation_off=n.buses.at[bus, "substation_off"],
            overwrite=True,
        )

        #connect the industry processes CC to the carbon cluster co2 bus

        link_name = f'{node} process emissions CC"

        n.add(
            "Link",
            name=link_name,
            bus0=n.links.at[link_name, "bus0"],
            bus1=n.links.at[link_name, "bus1"],
            bus2=bus + " cluster",
            bus3=n.links.at[link_name, "bus3"],
            bus4=n.links.at[link_name, "bus4"],
            p_nom_extendable=n.links.at[link_name, "p_nom_extendable"],
            p_min_pu=n.links.at[link_name, "p_min_pu"],
            carrier=n.links.at[link_name, "carrier"],
            efficiency=n.links.at[link_name, "efficiency"],
            efficiency2=n.links.at[link_name, "efficiency2"],
            efficiency3=n.links.at[link_name, "efficiency3"],
            efficiency4=n.links.at[link_name, "efficiency4"],
            capital_cost=n.links.at[link_name, "capital_cost"],
            marginal_cost=n.links.at[link_name, "marginal_cost"],
            lifetime=n.links.at[link_name, "lifetime"],
            reversed=False,
            overwrite=True,
        )


        link_name = f'{node} gas for industry CC"

        n.add(
            "Link",
            name=link_name,
            bus0=n.links.at[link_name, "bus0"],
            bus1=n.links.at[link_name, "bus1"],
            bus2=n.links.at[link_name, "bus2"],
            bus3=bus + " cluster",
            bus4=n.links.at[link_name, "bus4"],
            p_nom_extendable=n.links.at[link_name, "p_nom_extendable"],
            p_min_pu=n.links.at[link_name, "p_min_pu"],
            carrier=n.links.at[link_name, "carrier"],
            efficiency=n.links.at[link_name, "efficiency"],
            efficiency2=n.links.at[link_name, "efficiency2"],
            efficiency3=n.links.at[link_name, "efficiency3"],
            efficiency4=n.links.at[link_name, "efficiency4"],
            capital_cost=n.links.at[link_name, "capital_cost"],
            marginal_cost=n.links.at[link_name, "marginal_cost"],
            lifetime=n.links.at[link_name, "lifetime"],
            reversed=False,
            overwrite=True,
        )

        link_name = f'{node} solid biomass for industry CC"

        n.add(
            "Link",
            name=link_name,
            bus0=n.links.at[link_name, "bus0"],
            bus1=n.links.at[link_name, "bus1"],
            bus2=n.links.at[link_name, "bus2"],
            bus3=bus + " cluster",
            bus4=n.links.at[link_name, "bus4"],
            p_nom_extendable=n.links.at[link_name, "p_nom_extendable"],
            p_min_pu=n.links.at[link_name, "p_min_pu"],
            carrier=n.links.at[link_name, "carrier"],
            efficiency=n.links.at[link_name, "efficiency"],
            efficiency2=n.links.at[link_name, "efficiency2"],
            efficiency3=n.links.at[link_name, "efficiency3"],
            efficiency4=n.links.at[link_name, "efficiency4"],
            capital_cost=n.links.at[link_name, "capital_cost"],
            marginal_cost=n.links.at[link_name, "marginal_cost"],
            lifetime=n.links.at[link_name, "lifetime"],
            reversed=False,
            overwrite=True,
        )

    return n

def assign_cluster_generators_and_electricity_buses_to_carbon_clusters(n, cluster_size, cluster_cost_reduction, renewables, nodes_with_clusters):
    
    nodes_renewables_cf = {}                #dictionary of dataframes by node and renewable type, sorting the generators by average capacity factor (ascending order)
    clusters_generators={}                      #dictionary of dataframes by node and renewable type, containing the generators assigned to the cluster  

    for node in nodes_with_clusters.copy():

        insufficient_generators = False  #if one node doesn't have enough renewable capacity even for one of the renewable in renewables, then the cluster cannot be created and we skip to the next node. 

        for renewable in renewables:

            nodes_renewables_cf[(node, renewable)] = pd.DataFrame(
                index=n.generators['p_nom_max'].loc[n.generators.index.astype(str).str.contains(f'{node}.*{renewable}$")].index,
                columns=["p_max_pu","p_nom_max"]  
            )

            clusters_generators[(node, renewable)] = pd.DataFrame()

            #we are considering the highest mean p_min_pu to determine the best generators per renewable available

            nodes_renewables_cf[(node, renewable)] ["p_max_pu"] = n.generators_t['p_max_pu'].loc[:, n.generators_t['p_max_pu'].columns.astype(str).str.contains(f"{node}.*{renewable}$")].mean()
            nodes_renewables_cf[(node, renewable)] ["p_nom_max"] = n.generators['p_nom_max'].loc[n.generators.index.astype(str).str.contains(f"{node}.*{renewable}$")]

            nodes_renewables_cf[(node, renewable)] = nodes_renewables_cf[(node, renewable)].sort_values("p_max_pu", ascending=True)

            number_gen=0
            

            while nodes_renewables_cf[(node, renewable)].iloc[0:number_gen+1]["p_nom_max"].sum() <= cluster_size:

                if number_gen >= len(nodes_renewables_cf[(node, renewable)]):
                    print(f"Not enough {renewable} generators at node {node} to reach cluster_size.")
                    nodes_with_clusters.remove(node)
                    insufficient_generators = True
                    break

                number_gen += 1

            if insufficient_generators:
                break


            clusters_generators[(node, renewable)]  = n.generators.loc[nodes_renewables_cf[(node, renewable)].index[0:number_gen+1]]
            remaining_capacity = nodes_renewables_cf[(node, renewable)].iloc[0:number_gen+1]["p_nom_max"].sum() - cluster_size

            
            clusters_generators[(node, renewable)].loc[clusters_generators[(node, renewable)].index[number_gen], "p_nom_max"] = cluster_size - clusters_generators[(node, renewable)].loc[clusters_generators[(node, renewable)].index[0:number_gen],"p_nom_max"].sum()


            for idx in clusters_generators[(node, renewable)].index:

                ### Electricity bus and generators ###

                if not n.buses.index.str.contains(f"{clusters_generators[(node, renewable)].loc[idx].bus + ' cluster'}$").any():
        
                    n.add(
                        "Bus",
                        name=clusters_generators[(node, renewable)].loc[idx].bus + " cluster",
                        v_nom=n.buses.at[clusters_generators[(node, renewable)].loc[idx].bus, "v_nom"],
                        x=n.buses.at[clusters_generators[(node, renewable)].loc[idx].bus, "x"],
                        y=n.buses.at[clusters_generators[(node, renewable)].loc[idx].bus, "y"],
                        unit=n.buses.at[clusters_generators[(node, renewable)].loc[idx].bus, "unit"],
                        location=n.buses.at[clusters_generators[(node, renewable)].loc[idx].bus, "location"],
                        country=n.buses.at[clusters_generators[(node, renewable)].loc[idx].bus, "country"],
                        carrier=n.buses.at[clusters_generators[(node, renewable)].loc[idx].bus, "carrier"],
                        control=n.buses.at[clusters_generators[(node, renewable)].loc[idx].bus, "control"],
                        substation_lv=n.buses.at[clusters_generators[(node, renewable)].loc[idx].bus, "substation_lv"],
                        substation_off=n.buses.at[clusters_generators[(node, renewable)].loc[idx].bus, "substation_off"],
                    )

                n.add(
                    "Generator",
                    name=clusters_generators[(node, renewable)].loc[idx].name + " cluster",
                    bus=clusters_generators[(node, renewable)].loc[idx].bus + " cluster",
                    carrier=clusters_generators[(node, renewable)].loc[idx].carrier,
                    p_nom_max=clusters_generators[(node, renewable)].loc[idx].p_nom_max,
                    p_max_pu=clusters_generators[(node, renewable)].loc[idx].p_max_pu,
                    marginal_cost=clusters_generators[(node, renewable)].loc[idx].marginal_cost*(1-cluster_cost_reduction),
                    capital_cost=clusters_generators[(node, renewable)].loc[idx].capital_cost*(1-cluster_cost_reduction),
                    efficiency=clusters_generators[(node, renewable)].loc[idx].efficiency,
                    location=clusters_generators[(node, renewable)].loc[idx].location,
                    unit=clusters_generators[(node, renewable)].loc[idx].unit,
                    p_nom_extendable=True,
                    overwrite=True,)


                
                
                n.generators_t['p_max_pu'][clusters_generators[(node, renewable)].loc[idx].name + " cluster"] = n.generators_t['p_max_pu'][clusters_generators[(node, renewable)].loc[idx].name]


                ### H2 bus ##

                if not n.buses.index.str.contains(f"{clusters_generators[(node, renewable)].loc[idx].bus + ' H2 cluster'}$").any():

                    n.add(
                        "Bus",
                        name=clusters_generators[(node, renewable)].loc[idx].bus + " H2 cluster",
                        v_nom=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus + ' H2'}", "v_nom"],
                        x=n.buses.at[f"{clusters_generators[(node,renewable)].loc[idx].bus + ' H2'}", "x"],
                        y=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus + ' H2'}", "y"],
                        unit=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus + ' H2'}", "unit"],
                        location=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus + ' H2'}", "location"],
                        country=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus + ' H2'}", "country"],
                        carrier=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus + ' H2'}", "carrier"],
                        control=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus + ' H2'}", "control"],
                        substation_lv=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus + ' H2'}", "substation_lv"],
                        substation_off=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus + ' H2'}", "substation_off"],
                    )

                ### methanol bus ###

                if not n.buses.index.str.contains(f"{clusters_generators[(node, renewable)].loc[idx].bus + ' methanol cluster'}$").any():

                    n.add(
                        "Bus",
                        name=clusters_generators[(node, renewable)].loc[idx].bus + " methanol cluster",
                        v_nom=n.buses.at["EU methanol", "v_nom"],
                        x=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus}", "x"],
                        y=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus }", "y"],
                        unit=n.buses.at["EU methanol", "unit"],
                        location=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus  }", "location"],
                        country=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus }", "country"],
                        carrier=n.buses.at["EU methanol", "carrier"],
                        control=n.buses.at["EU methanol", "control"],
                        substation_lv=n.buses.at["EU methanol", "substation_lv"],
                        substation_off=n.buses.at["EU methanol", "substation_off"],
                    )
                
                ### Batteries bus ###

                if not n.buses.index.str.contains(f"{clusters_generators[(node, renewable)].loc[idx].bus + ' battery cluster'}$").any():

                    n.add(
                        "Bus",
                        name=clusters_generators[(node, renewable)].loc[idx].bus + " battery cluster",
                        v_nom=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus + ' battery'}", "v_nom"],
                        x=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus + ' battery'}", "x"],
                        y=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus + ' battery'}", "y"],
                        unit=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus + ' battery'}", "unit"],
                        location=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus + ' battery'}", "location"],
                        country=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus + ' battery'}", "country"],
                        carrier=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus + ' battery'}", "carrier"],
                        control=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus + ' battery'}", "control"],
                        substation_lv=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus + ' battery'}", "substation_lv"],
                        substation_off=n.buses.at[f"{clusters_generators[(node, renewable)].loc[idx].bus + ' battery'}", "substation_off"],
                    )



                if idx == nodes_renewables_cf[(node, renewable)].iloc[number_gen].name:
                    n.generators.loc[n.generators.index == idx, "p_nom_max"] = nodes_renewables_cf[(node, renewable)].iloc[0:number_gen+1]["p_nom_max"].sum() - cluster_size

                    print(f"Residual capacity of generator {clusters_generators[(node, renewable)].loc[idx].name} is {n.generators.loc[n.generators.index == idx, 'p_nom_max']} MW")
                
                else:


                    n.remove(
                            "Generator",
                            name=clusters_generators[(node, renewable)].loc[idx].name,
                    )



            

    return n

def add_cluster_links(n, nodes_with_clusters, cluster_cost_reduction, ongrid, cluster_seq):

    for node in nodes_with_clusters:

        print(node)


        ### H2 Electrolysis ###

        link_name = f"{node} H2 Electrolysis"

        n.add(
            "Link",
            name=link_name + " cluster",
            bus0=n.links.at[link_name, "bus0"] + " cluster",
            bus1=n.links.at[link_name, "bus1"] + " cluster",
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
        cluster_methanol_bus_name = f"{node} methanol cluster"

        n.add(
            "Link",
            name=link_name + " cluster",
            bus0=n.links.at[link_name, "bus0"] + " cluster",
            bus1=cluster_methanol_bus_name,
            bus2=n.links.at[link_name, "bus2"] + " cluster",
            bus3=n.links.at[link_name, "bus3"] + " cluster",
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
            name=f"{node} methanol cluster",
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

        if cluster_seq==True:

            ###Sequestration link

            link_name= f"{node} co2 sequestered cluster"

            n.add(
                "Link",
                name=link_name,
                bus0=f"{node} co2 stored cluster",
                bus1=n.links.at[f"{node} co2 sequestered", "bus1"],
                carrier=n.links.at[f"{node} co2 sequestered", "carrier"],  
                p_nom_extendable=True,
                efficiency=1.0,
                capital_cost=n.links.at[f"{node} co2 sequestered", "capital_cost"],
                marginal_cost=n.links.at[f"{node} co2 sequestered", "marginal_cost"],
                reversed=False,
                overwrite=True,
            )

            link_name= f"{node} co2 sequestered cluster"

            n.add(
                "Link",
                name=link_name,
                bus0=f"{node} co2 stored cluster",
                bus1=f"{node} co2 stored",
                carrier=n.buses.at[f"{node} co2 stored", "carrier"],  
                p_nom_extendable=True,
                efficiency=1.0,
                capital_cost=0.000000000001,
                marginal_cost=0.000000000001,
                reversed=False,
                overwrite=True,
            )


        if ongrid==True :

            ### Electricity connection to grid ###

            link_name = f"{node} electricity cluster"
            
            n.add(
                "Link",
                name=link_name,
                bus0=f"{node} cluster",
                bus1=f"{node}",
                carrier=n.buses.at[f"{node}", "carrier"],  
                p_nom_extendable=True,
                efficiency=1.0,
                capital_cost=0.0,
                marginal_cost=0.0,
                reversed=False,
                overwrite=True,
            )

            link_name = f"{node} electricity cluster back"
            n.add(
                "Link",
                name=link_name,
                bus0=f"{node}",
                bus1=f"{node} cluster",
                carrier=n.buses.at[f"{node}", "carrier"],  
                p_nom_extendable=True,
                efficiency=1.0,
                capital_cost=0.0,
                marginal_cost=0.0,
                reversed=True,
                overwrite=True,
            )

        else:
            if f"{node} cluster electricity" in n.links.index:
                n.remove(
                    "Link",
                    name=f"{node} cluster electricity",
                )
            if f"{node} cluster electricity back" in n.links.index:
                n.remove(
                    "Link",
                    name=f"{node} cluster electricity back",
                )

    return n


def add_cluster_storages(n, nodes_with_clusters, cluster_cost_reduction):

    for node in nodes_with_clusters:

        link_name = f"{node} H2 Store"

    
        n.add("Store",
            name=link_name + " cluster",
            bus=n.stores.at[link_name, "bus"] + " cluster",
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
                name=link_name + " charger cluster",
                bus0=f"{node} cluster",
                bus1=f"{node} battery cluster",
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
                name=link_name + " discharger cluster",
                bus0=f"{node} battery cluster",
                bus1=f"{node} cluster",
                carrier=n.buses.at[link_name, "carrier"],
                p_nom_extendable=True,
                efficiency=1.0,
                capital_cost=0.0,
                marginal_cost=0.0,
                reversed=True,
                overwrite=True,
            )

        n.add("Store",
            name=link_name + " cluster" ,
            bus=f"{node} battery cluster",
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
            name=link_name + " cluster" ,
            bus=f"{node} co2 stored cluster",
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
