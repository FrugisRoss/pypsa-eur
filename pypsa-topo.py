#%%

import pypsatopo
import pypsa
import yaml
from pathlib import Path
import geopandas as gpd


#%%
fn_50MW = r'results/Iberic100_2035_10ccslimit_noFR_24h/all/networks/base_s_40__24h_2035.nc'
fn_base = r'results/24h_test/Iberic100_2035_10ccslimit_noFR_24h/all/networks/base_s_40__24h_2035.nc'
config = yaml.safe_load(Path("config/config.iberic100_2050_modco2_3h.yaml").read_text())
config_plotting = yaml.safe_load(Path("config/plotting.default.yaml").read_text())                        
regions = gpd.read_file(r'resources/Iberic100_2035_10ccslimit_noFR_24h/all/regions_onshore_base_s_40.geojson').set_index("name")

n_50MW = pypsa.Network(fn_50MW)
n_base = pypsa.Network(fn_base)



#%%

pypsatopo.generate(n_50MW)
# %%
