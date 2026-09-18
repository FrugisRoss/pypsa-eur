#%%

import copy
import pypsa
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Polygon
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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import pypsa
from packaging.version import Version, parse
from pypsa.plot import (
    add_legend_patches,
    add_legend_semicircles,
)
from pypsa.statistics import get_transmission_carriers
import geopandas as gdp
from scripts.plot_power_network import load_projection


label_to_colors = {
    'methanolisation': "#2df7d6",
    'solid biomass biomass-to-methanol': "#36ad0e",
    'H2 Electrolysis': "#187878",
    'solar rooftop': "#ffe204",
    'solar': "#f47b0a",
    'solar-hsat': "#e32f0b",
    'urban central solar thermal collector': "#3e1c04",
    'urban decentral solar thermal collector': '#87402e',
    'rural solar thermal collector': '#f5a4b4',
    'Net Balance': "#FB0202",
    'onwind': "#0281d6",
    'Fischer-Tropsch': "#4e42f5",
    'Sabatier': '#ebb028',

    # Renewable cluster variants
    'methanolisation renewable cluster': '#EB28B0',
    'H2 Electrolysis renewable cluster': "#473BF5",
    'solar-hsat renewable cluster': "#870000",
    'solar renewable cluster': "#bff542",
    'battery charger renewable cluster': "#193ade",
    'battery discharger renewable cluster': "#0d0f5c",
    'electricity renewable cluster': "#BEBBFA",
    'electricity renewable cluster back': "#889899",
    'H2 Store renewable cluster charge': "#59786d",
    'H2 Store renewable cluster discharge': "#2A483B",
    'onwind renewable cluster': "#8ad0ff",
    'Fischer-Tropsch renewable cluster': "#c99799",
    'Sabatier renewable cluster': '#debf12',
    'urban central DAC renewable cluster': "#3e1c04",
    'electricity renewable cluster both': "#BEBBFA",

    'DAC': "#b93af0",
    'SMR CC': "#6b3a1f",
    'biogas to gas CC': "#8b4513",
    'gas for industry CC': "#5c3317",
    'process emissions CC': "#0d0d0d",
    'solid biomass for industry CC': "#5a6b23",
    'urban central gas CHP CC': "#7a4a21",
    'urban central solid biomass CHP CC': "#4f6b2f",
}

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
        "BOTHcap": find_wildcard_value(name, "BOTHcap"),
    }

#%%


def co2_captured_cluster(n) -> float:
    """
    CO2 captured into the industrial cluster's own CO2 store [Mtons/year]
    for a single loaded network.

    On the "... DAC renewable cluster" links, bus3 is the cluster's own CO2
    store, so p3 (weighted by snapshot_weightings, summed over snapshots) is
    the annual CO2 mass captured into the cluster.
    """
    mask = n.links_t.p3.columns.str.contains("DAC renewable cluster")
    weights = n.snapshot_weightings.generators
    captured_tons = n.links_t.p3.loc[:, mask].multiply(weights, axis=0).sum().sum()
    return abs(captured_tons) / 1e6


def co2_marginal_price(n) -> float:
    """
    CO2 marginal price [euro/ton] for a single loaded network, taken from the
    shadow price (mu) of the CO2Limit global constraint.
    """
    mask = n.global_constraints.index.str.contains("CO2Limit")
    return abs(n.global_constraints.loc[mask, "mu"].iloc[0])


def renewable_capacity_cluster(n) -> float:
    """
    Total optimised renewable generator capacity installed in the industrial
    clusters [MW] for a single loaded network.
    """
    mask = n.generators.index.str.contains("renewable cluster")
    return n.generators.loc[mask, "p_nom_opt"].sum()


# Column name -> function computing that column from a single loaded network.
NETWORK_METRICS = {
    "CO₂ Captured Cluster [Mtons/year]": co2_captured_cluster,
    "CO₂ Price [€/ton]": co2_marginal_price,
    "Renewable Capacity Cluster [MW]": renewable_capacity_cluster,
}


def add_network_metrics(df: pd.DataFrame, metrics: dict = NETWORK_METRICS) -> pd.DataFrame:
    """
    Load each network from its path exactly once and add one column per entry
    in ``metrics`` (mapping column name -> function(n) -> scalar).
    """
    df = df.copy()
    cols = {name: [] for name in metrics}
    for path in df["path"]:
        n = pypsa.Network(str(path))
        for name, fn in metrics.items():
            cols[name].append(fn(n))
    for name, vals in cols.items():
        df[name] = vals
    return df


def plot_heatmap_buycap_vs_cr(
    df: pd.DataFrame,
    sellcap,
    bothcap,
    value_col,
    cmap="Purples",
    cr_range=None,
    buycap_range=None,
    title="CO₂ Captured in Cluster [Mtons/year]",
    main_title=None,
):
    """
    Filter df to rows where SELLcap == sellcap and BOTHcap == bothcap, then pivot on CR (x-axis)
    and BUYcap (y-axis) and plot a heatmap of the column ``value_col``.

    cr_range and buycap_range are optional (min, max) tuples restricting
    which CR / BUYcap values are included (default: all available values).
    """

    df = df.drop_duplicates(subset=["CR", "BUYcap", "SELLcap", "BOTHcap"], keep="first")
    df = df.loc[df.SELLcap == sellcap]
    df = df.loc[df.BOTHcap == bothcap]
    if cr_range is not None:
        df = df.loc[df.CR.between(cr_range[0], cr_range[1])]
    if buycap_range is not None:
        df = df.loc[df.BUYcap.between(buycap_range[0], buycap_range[1])]

    pivot = df.pivot(
        index="BUYcap", columns="CR", values=value_col
    ).sort_index(axis=0, ascending=True).sort_index(axis=1, ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel("CR")
    ax.set_ylabel("BUYcap")
    ax.set_xticklabels([f"{float(c):.0%}" for c in pivot.columns])
    ax.set_yticklabels([f"{float(i):.0%}" for i in pivot.index])
    ax.collections[0].colorbar.ax.set_ylabel(title, rotation=270, labelpad=15, va="bottom")
    fig.tight_layout()

    if main_title:
        fig.suptitle(main_title, fontsize=14, y=0.95)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

    return fig, ax


def plot_heatmap_sellcap_vs_cr(
    df: pd.DataFrame,
    buycap,
    bothcap,
    value_col,
    cmap="Purples",
    cr_range=None,
    sellcap_range=None,
    title="CO₂ Captured in Cluster [Mtons/year]",
    main_title=None,
):
    """
    Filter df to rows where BUYcap == buycap and BOTHcap == bothcap, then pivot on CR (x-axis)
    and SELLcap (y-axis) and plot a heatmap of the column ``value_col``.

    cr_range and sellcap_range are optional (min, max) tuples restricting
    which CR / SELLcap values are included (default: all available values).
    """

    df = df.drop_duplicates(subset=["CR", "BUYcap", "SELLcap", "BOTHcap"], keep="first")
    df = df.loc[df.BUYcap == buycap]
    df = df.loc[df.BOTHcap == bothcap]

    if cr_range is not None:
        df = df.loc[df.CR.between(cr_range[0], cr_range[1])]
    if sellcap_range is not None:
        df = df.loc[df.SELLcap.between(sellcap_range[0], sellcap_range[1])]
    pivot = df.pivot(
        index="SELLcap", columns="CR", values=value_col
    ).sort_index(axis=0, ascending=True).sort_index(axis=1, ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel("CR")
    ax.set_ylabel("SELLcap")
    ax.set_xticklabels([f"{float(c):.0%}" for c in pivot.columns])
    ax.set_yticklabels([f"{float(i):.0%}" for i in pivot.index])
    ax.collections[0].colorbar.ax.set_ylabel(title, rotation=270, labelpad=15, va="bottom")
    fig.tight_layout()

    if main_title:
        fig.suptitle(main_title, fontsize=14, y=0.95)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

    return fig, ax

def plot_heatmap_bothcap_vs_cr(
    df: pd.DataFrame,
    buycap,
    sellcap,
    value_col,
    cmap="Purples",
    cr_range=None,
    bothcap_range=None,
    title="CO₂ Captured in Cluster [Mtons/year]",
    main_title=None,
):
    """
    Filter df to rows where BUYcap == buycap and SELLcap == sellcap,
    then pivot on CR (x-axis) and BOTHcap (y-axis) and plot a heatmap
    of the column ``value_col``.

    cr_range and bothcap_range are optional (min, max) tuples restricting
    which CR / BOTHcap values are included (default: all available values).
    """

    df = df.drop_duplicates(subset=["CR", "BUYcap", "SELLcap", "BOTHcap"], keep="first")
    df = df.loc[df.BUYcap == buycap]
    df = df.loc[df.SELLcap == sellcap]

    if cr_range is not None:
        df = df.loc[df.CR.between(cr_range[0], cr_range[1])]
    if bothcap_range is not None:
        df = df.loc[df.BOTHcap.between(bothcap_range[0], bothcap_range[1])]

    pivot = df.pivot(
        index="BOTHcap", columns="CR", values=value_col
    ).sort_index(axis=0, ascending=True).sort_index(axis=1, ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel("CR")
    ax.set_ylabel("BOTHcap")
    ax.set_xticklabels([f"{float(c):.0%}" for c in pivot.columns])
    ax.set_yticklabels([f"{float(i):.0%}" for i in pivot.index])
    ax.collections[0].colorbar.ax.set_ylabel(title, rotation=270, labelpad=15, va="bottom")
    fig.tight_layout()

    if main_title:
        fig.suptitle(main_title, fontsize=14, y=0.95)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

    return fig, ax


def load_metrics_df(networks_folder, region: str = None) -> pd.DataFrame:
    """
    Build the wildcard/metrics DataFrame for one region: parse the CR/BUYcap/
    SELLcap/BOTHcap wildcards out of every network file in ``networks_folder``
    and add the NETWORK_METRICS columns. Each network is loaded exactly once.

    If ``region`` is given, it is added as a "region" column so several
    regions' DataFrames can be concatenated and later told apart.
    """
    records = []
    for path in sorted(Path(networks_folder).glob("*.nc")):
        wc = parse_wildcards(path)
        records.append({**wc, "path": path})
    df = pd.DataFrame(records)
    df = add_network_metrics(df)
    if region is not None:
        df["region"] = region
    return df


def plot_heatmap_two_regions(
    df: pd.DataFrame,
    region1,
    region2,
    axis,
    fixed,
    value_col,
    title1,
    title2,
    cmap="Purples",
    cr_range=None,
    axis_range=None,
    suptitle=None,
    main_title=None,
    region_col="region",
):
    """
    Build the same heatmap (``axis`` vs CR, for ``value_col``) for two regions
    and plot them side by side in one figure sharing a single color scale and
    colorbar.

    df: combined metrics DataFrame (e.g. built with load_metrics_df(..., region=...)
        for each region and concatenated) already containing ``region_col``.
    region1 / region2: the values of ``region_col`` identifying each region.
    axis: which wildcard varies on the y-axis together with CR
        ("BUYcap", "SELLcap" or "BOTHcap").
    fixed: dict of the other two wildcards' fixed values, e.g.
        {"SELLcap": 0, "BOTHcap": 0} when axis="BUYcap".
    """
    wildcard_cols = ["CR", "BUYcap", "SELLcap", "BOTHcap"]

    def build_pivot(region_value):
        d = df.loc[df[region_col] == region_value]
        d = d.drop_duplicates(subset=wildcard_cols, keep="first")
        for col, val in fixed.items():
            d = d.loc[d[col] == val]
        if cr_range is not None:
            d = d.loc[d.CR.between(cr_range[0], cr_range[1])]
        if axis_range is not None:
            d = d.loc[d[axis].between(axis_range[0], axis_range[1])]
        return (
            d.pivot(index=axis, columns="CR", values=value_col)
            .sort_index(axis=0, ascending=True)
            .sort_index(axis=1, ascending=True)
        )

    pivot1 = build_pivot(region1)
    pivot2 = build_pivot(region2)

    vmin = min(pivot1.min().min(), pivot2.min().min())
    vmax = max(pivot1.max().max(), pivot2.max().max())

    def pct_labels(values):
        return [f"{float(v):.0%}" for v in values]

    fig, (ax1, ax2, cax) = plt.subplots(
        1, 3, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1, 0.05]}
    )

    sns.heatmap(pivot1, annot=True, fmt=".2f", cmap=cmap, ax=ax1, vmin=vmin, vmax=vmax, cbar=False)
    ax1.invert_yaxis()
    ax1.set_xlabel("CR")
    ax1.set_ylabel(axis)
    ax1.set_title(title1)
    ax1.set_xticklabels(pct_labels(pivot1.columns))
    ax1.set_yticklabels(pct_labels(pivot1.index))

    sns.heatmap(
        pivot2, annot=True, fmt=".2f", cmap=cmap, ax=ax2, vmin=vmin, vmax=vmax,
        cbar=True, cbar_ax=cax,
    )
    ax2.invert_yaxis()
    ax2.set_xlabel("CR")
    ax2.set_ylabel(axis)
    ax2.set_title(title2)
    ax2.set_xticklabels(pct_labels(pivot2.columns))
    ax2.set_yticklabels(pct_labels(pivot2.index))

    if suptitle:
        cax.set_ylabel(suptitle, rotation=270, labelpad=15, va="bottom")

    fig.tight_layout()

    if main_title:
        fig.suptitle(main_title, fontsize=14, y=0.95)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

    return fig, (ax1, ax2, cax)


#%%
# Build the combined metrics DataFrame once: each network (both regions) is
# loaded from disk exactly once, and every plot below (single-region and
# combined) reuses this in-memory DataFrame instead of reloading networks.

iberian_networks_folder = r"results/Iberic40_2035_industrial_clusters/all/networks"
nordic_networks_folder = r"results/Noridcs100_2035_industrial_clusters/all/networks"

config_plotting = yaml.safe_load(Path("config/plotting.default.yaml").read_text())
regions_iberian = gdp.read_file(r'resources/Iberic40_2035_industrial_clusters/all/regions_onshore_base_s_40.geojson').set_index("name")
regions_nordic = gdp.read_file(r'resources/Noridcs100_2035_industrial_clusters/all/regions_onshore_base_s_100.geojson').set_index("name")

REGION_IBERIAN = "Iberian Peninsula"
REGION_NORDIC = "Nordic Countries"

df_iberian = load_metrics_df(iberian_networks_folder, region=REGION_IBERIAN)
df_nordic = load_metrics_df(nordic_networks_folder, region=REGION_NORDIC)
df_all = pd.concat([df_iberian, df_nordic], ignore_index=True)

#%%
#Iberian Peninsula

df = df_iberian

#%%

fig, ax = plot_heatmap_buycap_vs_cr(df, 0, 0, "CO₂ Captured Cluster [Mtons/year]", "Purples", cr_range=None, buycap_range=(0, 1), title="CO₂ Captured in Renewable Clusters [Mtons/year]", main_title="Iberian Peninsula - SELLcap=0%, BOTHcap=0%")
fig, ax = plot_heatmap_sellcap_vs_cr(df, 0, 0, "CO₂ Captured Cluster [Mtons/year]", "Purples", cr_range=None, sellcap_range=(0, 0.4), title="CO₂ Captured in Renewable Clusters [Mtons/year]", main_title="Iberian Peninsula - BUYcap=0%, BOTHcap=0%")
# fig, ax = plot_heatmap_bothcap_vs_cr(df, 0, 0, "CO₂ Captured Cluster [Mtons/year]", "Purples", cr_range=None, bothcap_range=(0, 0.5), title="CO₂ Captured in Renewable Clusters [Mtons/year]", main_title="Iberian Peninsula 2035 - BUYcap=0%, SELLcap=0%")

fig, ax = plot_heatmap_buycap_vs_cr(df, 0, 0, "CO₂ Price [€/ton]", "Reds", cr_range=None, buycap_range=(0, 1), title="CO₂ Marginal Price [€/ton]", main_title="Iberian Peninsula - SELLcap=0%, BOTHcap=0%")
fig, ax = plot_heatmap_sellcap_vs_cr(df, 0, 0, "CO₂ Price [€/ton]", "Reds", cr_range=None, sellcap_range=(0, 0.4), title="CO₂ Marginal Price [€/ton]", main_title="Iberian Peninsula - BUYcap=0%, BOTHcap=0%")
# fig, ax = plot_heatmap_bothcap_vs_cr(df, 0, 0, "CO₂ Price [€/ton]", "Reds", cr_range=None, bothcap_range=(0, 0.5), title="CO₂ Marginal Price [€/ton]", main_title="Iberian Peninsula 2035 - BUYcap=0%, SELLcap=0%")

fig, ax = plot_heatmap_buycap_vs_cr(df, 0, 0, "Renewable Capacity Cluster [MW]", "Greens", cr_range=None, buycap_range=(0, 1), title="Renewable Capacity in Clusters [MW]", main_title="Iberian Peninsula - SELLcap=0%, BOTHcap=0%")
fig, ax = plot_heatmap_sellcap_vs_cr(df, 0, 0, "Renewable Capacity Cluster [MW]", "Greens", cr_range=None, sellcap_range=(0, 0.4), title="Renewable Capacity in Clusters [MW]", main_title="Iberian Peninsula - BUYcap=0%, BOTHcap=0%")
# fig, ax = plot_heatmap_bothcap_vs_cr(df, 0, 0, "Renewable Capacity Cluster [MW]", "Greens", cr_range=None, bothcap_range=(0, 0.5), title="Renewable Capacity in Clusters [MW]", main_title="Iberian Peninsula 2035 - BUYcap=0%, SELLcap=0%")


# %%
#Noridc Countries

df = df_nordic

#%%
fig, ax = plot_heatmap_buycap_vs_cr(df, 0, 0, "CO₂ Captured Cluster [Mtons/year]", "Purples", cr_range=None, buycap_range=(0, 1), title="CO₂ Captured in Renewable Clusters [Mtons/year]", main_title="Nordics Countries - SELLcap=0%")
fig, ax = plot_heatmap_sellcap_vs_cr(df, 0, 0, "CO₂ Captured Cluster [Mtons/year]", "Purples", cr_range=None, sellcap_range=(0, 0.4), title="CO₂ Captured in Renewable Clusters [Mtons/year]", main_title="Nordic Countries - BUYcap=0%")
# fig, ax = plot_heatmap_bothcap_vs_cr(df, 0, 0, "CO₂ Captured Cluster [Mtons/year]", "Purples", cr_range=None, bothcap_range=(0, 0.5), title="CO₂ Captured in Renewable Clusters [Mtons/year]", main_title="Nordics 2035 - BUYcap=0%, SELLcap=0%")

fig, ax = plot_heatmap_buycap_vs_cr(df, 0, 0, "CO₂ Price [€/ton]", "Reds", cr_range=None, buycap_range=(0, 1), title="CO₂ Marginal Price [€/ton]", main_title="Nordic Countries- SELLcap=0%")
fig, ax = plot_heatmap_sellcap_vs_cr(df, 0, 0, "CO₂ Price [€/ton]", "Reds", cr_range=None, sellcap_range=(0, 0.4), title="CO₂ Marginal Price [€/ton]", main_title="Nordic Countries - BUYcap=0%")
# fig, ax = plot_heatmap_bothcap_vs_cr(df, 0, 0, "CO₂ Price [€/ton]", "Reds", cr_range=None, bothcap_range=(0, 0.5), title="CO₂ Marginal Price [€/ton]", main_title="Nordics 2035 - BUYcap=0%, SELLcap=0%")

fig, ax = plot_heatmap_buycap_vs_cr(df, 0, 0, "Renewable Capacity Cluster [MW]", "Greens", cr_range=None, buycap_range=(0, 1), title="Renewable Capacity in Clusters [MW]", main_title="Nordic Countries - SELLcap=0%")
fig, ax = plot_heatmap_sellcap_vs_cr(df, 0, 0, "Renewable Capacity Cluster [MW]", "Greens", cr_range=None, sellcap_range=(0, 0.4), title="Renewable Capacity in Clusters [MW]", main_title="Nordics Countries - BUYcap=0%")
# fig, ax = plot_heatmap_bothcap_vs_cr(df, 0, 0, "Renewable Capacity Cluster [MW]", "Greens", cr_range=None, bothcap_range=(0, 0.5), title="Renewable Capacity in Clusters [MW]", main_title="Nordics 2035 - BUYcap=0%, SELLcap=0%")

#%%
# Combined Iberian Peninsula + Nordics heatmaps, one figure per metric, shared color scale/legend.
# Reuses df_all (built once above) - no network reloading here.

for value_col, cmap in [
    ("CO₂ Captured Cluster [Mtons/year]", "Purples"),
    ("CO₂ Price [€/ton]", "Reds"),
    ("Renewable Capacity Cluster [MW]", "Greens"),
]:
    fig, axes = plot_heatmap_two_regions(
        df_all,
        REGION_IBERIAN,
        REGION_NORDIC,
        axis="BUYcap",
        fixed={"SELLcap": 0, "BOTHcap": 0},
        value_col=value_col,
        title1=REGION_IBERIAN,
        title2=REGION_NORDIC,
        cmap=cmap,
        cr_range=None,
        axis_range=(0, 1),
        suptitle=value_col,
        main_title="SELLcap=0%",
    )

    fig, axes = plot_heatmap_two_regions(
        df_all,
        REGION_IBERIAN,
        REGION_NORDIC,
        axis="SELLcap",
        fixed={"BUYcap": 0, "BOTHcap": 0},
        value_col=value_col,
        title1=REGION_IBERIAN,
        title2=REGION_NORDIC,
        cmap=cmap,
        cr_range=None,
        axis_range=(0, 0.4),
        suptitle=value_col,
        main_title="BUYcap=0%",
    )

#%%

