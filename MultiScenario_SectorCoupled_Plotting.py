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
import re





def find_wildcard_value(name: str, key: str) -> float:
    """
    Return the numeric value of a '{key}+<value>' wildcard in a filename,
    or 0.0 if that wildcard is not present (e.g. BUYcap when onBUY is off).
    """
    m = re.search(rf"{key}\+([\dp]+)", name)
    if m is None:
        return 0.0
    return float(m.group(1).replace("p", "."))


def parse_wildcards(path: Path) -> dict:
    name = path.name
    return {
        "CR": find_wildcard_value(name, "CR"),
        "BUYcap": find_wildcard_value(name, "BUYcap"),
        "SELLcap": find_wildcard_value(name, "SELLcap"),
    }

#%%


def add_co2_captured(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load each network from its path and add a column with the CO2 captured
    into the industrial cluster's CO2 store [Mtons/year].

    On the "... DAC renewable cluster" links, bus3 is the cluster's own CO2
    store, so p3 (weighted by snapshot_weightings, summed over snapshots) is
    the annual CO2 mass captured into the cluster.
    """
    df = df.copy()
    co2_captured = []
    for path in df["path"]:
        n = pypsa.Network(str(path))
        mask = n.links_t.p3.columns.str.contains("DAC renewable cluster")
        weights = n.snapshot_weightings.generators
        captured_tons = n.links_t.p3.loc[:, mask].multiply(weights, axis=0).sum().sum()
        co2_captured.append(abs(captured_tons) / 1e6)
    df["co2 captured cluster [Mtons/year]"] = co2_captured
    return df


def plot_co2_heatmap(df: pd.DataFrame, title="CO2 captured in cluster [Mtons/year]"):
    """
    Pivot df on CR (x-axis) and BUYcap (y-axis) and plot a heatmap of
    'co2 captured cluster [Mtons/year]'.
    """
    pivot = df.pivot(
        index="BUYcap", columns="CR", values="co2 captured cluster [Mtons/year]"
    ).sort_index(axis=0, ascending=True).sort_index(axis=1, ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Purples", ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel("CR")
    ax.set_ylabel("BUYcap")
    ax.set_title(title)
    return fig, ax
#%%
networks_folder=r"results/Iberic40_2035_industrial_clusters_8h/all/networks"


records = []
for path in sorted(Path(networks_folder).glob("*.nc")):
    wc = parse_wildcards(path)
    records.append({**wc, "path": path})

df = pd.DataFrame(records)

df = add_co2_captured(df)
fig, ax = plot_co2_heatmap(df,"CO2 captured in renewable clusters [Mtons/year] - Iberian Peninsula 2035")
# %%

networks_folder=r"results/Noridcs100_2035_industrial_clusters_8h/all/networks"


records = []
for path in sorted(Path(networks_folder).glob("*.nc")):
    wc = parse_wildcards(path)
    records.append({**wc, "path": path})

df = pd.DataFrame(records)

df = add_co2_captured(df)
fig, ax = plot_co2_heatmap(df,"CO2 captured in renewable clusters [Mtons/year] - Nordics 2035")

# %%
