#%%

import pypsa
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
from pypsa.plot import add_legend_lines, add_legend_patches, add_legend_semicircles
import numpy as np  
import pandas as pd
from pathlib import Path
import yaml
from matplotlib.colors import ListedColormap
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go



config = yaml.safe_load(Path("config/config.denmark.yaml").read_text())


#%%

# Scenarios 

# In scenarios_pointsources and scenarios_renewables the keys are numbers that indicate the cost reduction of capex and opex for the technologies inside the cluster with respect to the system-equivalent ones
# The values are the paths to the corresponding network files  

scenarios_pointsources = { '100% cost red': r'/Users/rofrug/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/First Year PhD/Pypsa/results/DK_test_2Feb2026/CO2_clusters/100/networks/base_s_2__12h_2050.nc',
                        '80% cost red': r'/Users/rofrug/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/First Year PhD/Pypsa/results/DK_test_2Feb2026/CO2_clusters/80/networks/base_s_2__12h_2050.nc',
                        '60% cost red': r'/Users/rofrug/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/First Year PhD/Pypsa/results/DK_test_2Feb2026/CO2_clusters/60/networks/base_s_2__12h_2050.nc',
                        '40% cost red': r'/Users/rofrug/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/First Year PhD/Pypsa/results/DK_test_2Feb2026/CO2_clusters/40/networks/base_s_2__12h_2050.nc',
                        '20% cost red': r'/Users/rofrug/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/First Year PhD/Pypsa/results/DK_test_2Feb2026/CO2_clusters/20/networks/base_s_2__12h_2050.nc',
                        '0% cost red': r'/Users/rofrug/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/First Year PhD/Pypsa/results/DK_test_2Feb2026/CO2_clusters/0/networks/base_s_2__12h_2050.nc'
                    }
scenarios_renewables = {'100% cost red': r'/Users/rofrug/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/First Year PhD/Pypsa/results/DK_test_2Feb2026/Renewable_clusters/100/networks/base_s_2__12h_2050.nc',
                        '80% cost red': r'/Users/rofrug/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/First Year PhD/Pypsa/results/DK_test_2Feb2026/Renewable_clusters/80/networks/base_s_2__12h_2050.nc',
                        '60% cost red': r'/Users/rofrug/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/First Year PhD/Pypsa/results/DK_test_2Feb2026/Renewable_clusters/60/networks/base_s_2__12h_2050.nc',
                        '40% cost red': r'/Users/rofrug/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/First Year PhD/Pypsa/results/DK_test_2Feb2026/Renewable_clusters/40/networks/base_s_2__12h_2050.nc',
                        '20% cost red': r'/Users/rofrug/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/First Year PhD/Pypsa/results/DK_test_2Feb2026/Renewable_clusters/20/networks/base_s_2__12h_2050.nc',
                        '0% cost red': r'/Users/rofrug/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/First Year PhD/Pypsa/results/DK_test_2Feb2026/Renewable_clusters/0/networks/base_s_2__12h_2050.nc'
                        }

scenarios= { 
            'Best Renewable Resources Based Clusters' : scenarios_renewables,
            'Point Source Based Clusters' : scenarios_pointsources,
            }


#%%

label_to_colors = {'methanolisation cluster':'#a442f5',
                    'methanolisation':"#401662",
                    'solid biomass biomass-to-methanol':"#8442f5",
                    'H2 Electrolysis cluster':"#42f5d1",
                    'H2 Electrolysis':"#187878",
                    'solar rooftop':"#ffe204",                     
                    'solar':"#f47b0a",                                   
                    'solar-hsat':"#e32f0b",                                 
                    'solar-hsat cluster':"#870000",                         
                    'solar cluster':"#bff542",                              
                    'urban central solar thermal collector':"#3e1c04",      
                    'urban decentral solar thermal collector':'#87402e',    
                    'rural solar thermal collector':'#f5a4b4',
                    'Net Balance':"#FB0202",
                    'battery charger cluster':"#193ade",
                    'battery discharger cluster':"#0d0f5c",
                    'electricity cluster':"#BEBBFA",
                    'electricity cluster back':"#889899",
                    'H2 Store cluster charge':"#59786d",
                    'H2 Store cluster discharge':"#2A483B",
                    'onwind cluster':"#8ad0ff",
                    'onwind':"#0281d6",

                   }



#%%

for scenario_type, scenario_dict in scenarios.items():

    for country in config ["countries"]:

        # Dictionaries to store data for all scenarios
        methanol_data = {}
        hydrogen_data = {}

        for scenario_name, scenario_path in scenario_dict.items():

            n=pypsa.Network(scenario_path)



            # Create a figure for each scenario type

            # Methanol
            methanol_technologies = n.links.loc[
                ((n.links["bus1"].str.contains("methanol cluster")) | 
                    (n.links["bus1"].str.contains("EU methanol"))) & 
                (n.links.index.str.contains(country)) & 
                (~n.links.index.str.contains("methanol cluster"))
            ].index
            
            methanol_production_t = (n.links_t["p1"][methanol_technologies]).abs() / 10**3
            tech = methanol_production_t.columns.str.slice(start=6)
            methanol_production_bytech = (
                methanol_production_t.groupby(tech, axis=1).sum()
            ).sum()
            
            methanol_data[scenario_name] = methanol_production_bytech
            
            # Hydrogen
            electrolyzer_technologies = n.links.loc[
                (n.links["carrier"] == "H2 Electrolysis") & 
                (n.links.index.str.contains(country))
            ].index
            
            tech = electrolyzer_technologies.str.slice(start=6)
            electrolyzers_production_t = (n.links_t["p1"][electrolyzer_technologies].abs()) / 10**3
            electrolyzers_production_bytech = (
                electrolyzers_production_t.groupby(tech, axis=1).sum()
            ).sum()
            
            hydrogen_data[scenario_name] = electrolyzers_production_bytech

        # Convert to DataFrames
        methanol_df = pd.DataFrame(methanol_data).T
        hydrogen_df = pd.DataFrame(hydrogen_data).T
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get all unique technologies from both dataframes
        all_techs = list(set(methanol_df.columns.tolist() + hydrogen_df.columns.tolist()))
        
        # Set up positions for bars
        n_scenarios = len(scenario_dict)
        x = np.arange(n_scenarios)
        width = 0.35
        
        scenario_names = list(scenario_dict.keys())

        # Plot stacked bars for Methanol
        bottom_meoh = np.zeros(n_scenarios)
        for tech in methanol_df.columns:
            values = methanol_df[tech].values
            color = label_to_colors.get(tech, '#808080')
            ax.bar(x - width/2, values, width, bottom=bottom_meoh, 
                label=tech, color=color, alpha=0.8)
            bottom_meoh += values
        
        # Plot stacked bars for Hydrogen
        bottom_h2 = np.zeros(n_scenarios)
        for tech in hydrogen_df.columns:
            values = hydrogen_df[tech].values
            color = label_to_colors.get(tech, '#808080')
            # Only add to legend if not already added from methanol
            if tech not in methanol_df.columns:
                ax.bar(x + width/2, values, width, bottom=bottom_h2, 
                    label=tech, color=color, alpha=0.8)
            else:
                ax.bar(x + width/2, values, width, bottom=bottom_h2, 
                    color=color, alpha=0.8)
            bottom_h2 += values
        
        # Customize plot
        ax.set_ylabel('Production [GWh]', fontsize=12, fontweight='bold')
        ax.set_title(f'{country} with {scenario_type}', fontsize=16, fontweight='bold')
        
        # Create custom x-tick labels with scenario names and product types
        ax.set_xticks(x - width/2)
        ax.set_xticklabels([''] * n_scenarios)  # Clear default labels
        
        # Add product type labels (Methanol/Hydrogen) below each bar
        for i, scenario in enumerate(scenario_names):
            ax.text(x[i] - width/2, -0.05, 'MeOH', 
                ha='center', va='top', transform=ax.get_xaxis_transform(), 
                fontsize=9, fontweight='bold')
            ax.text(x[i] + width/2, -0.05, 'H2', 
                ha='center', va='top', transform=ax.get_xaxis_transform(), 
                fontsize=9, fontweight='bold')
            # Add scenario name spanning both bars
            ax.text(x[i], -0.12, scenario, 
                ha='center', va='top', transform=ax.get_xaxis_transform(), 
                fontsize=11, fontweight='bold')
        
        ax.legend(title='Technologies', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

plt.show()
# %%
