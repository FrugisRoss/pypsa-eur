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
from matplotlib import font_manager
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


def parse_wildcards(path: Path) -> dict | None:
    """
    Return the wildcard values parsed from a network filename, or None if the
    network does not have the 'IC' (industrial cluster) wildcard.
    """
    name = path.name
    if re.search(r"(?<![A-Za-z])IC(?![A-Za-z])", name) is None:
        return None
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
    clusters [GW] for a single loaded network.
    """
    mask = n.generators.index.str.contains("renewable cluster")
    return n.generators.loc[mask, "p_nom_opt"].sum() / 1e3


def curtailment_share_cluster(n) -> float:
    """
    Share of the available renewable generation in the industrial clusters
    that is curtailed [%] for a single loaded network.

    Available generation is p_nom_opt * p_max_pu, actual generation is the
    dispatch p, both weighted by snapshot_weightings and summed over all
    cluster generators and snapshots.
    """
    generators = n.generators.index[
        n.generators.index.str.contains("renewable cluster")
    ]
    generators = generators[n.generators.loc[generators, "p_nom_opt"].gt(1)]

    if generators.empty:
        return 0.0

    weights = n.snapshot_weightings.generators

    available = (
        n.get_switchable_as_dense("Generator", "p_max_pu")[generators]
        .multiply(n.generators.loc[generators, "p_nom_opt"], axis=1)
        .multiply(weights, axis=0)
        .sum()
        .sum()
    )
    if available <= 0:
        return 0.0

    actual = n.generators_t.p[generators].multiply(weights, axis=0).sum().sum()

    curtailed_share = 1.0 - actual / available
    return float(np.clip(curtailed_share, 0.0, 1.0)) * 100


# Column name -> function computing that column from a single loaded network.
NETWORK_METRICS = {
    "CO₂ Captured Cluster [Mtons/year]": co2_captured_cluster,
    "CO₂ Price [€/ton]": co2_marginal_price,
    "Renewable Capacity Cluster [GW]": renewable_capacity_cluster,
    "Share of Curtailment in the Cluster[%]": curtailment_share_cluster,
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
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap=cmap, ax=ax, annot_kws={"fontsize": 12})
    ax.invert_yaxis()
    ax.set_xlabel("CR", fontsize=14)
    ax.set_ylabel("BUYcap", fontsize=14)
    ax.set_xticklabels([f"{float(c):.0%}" for c in pivot.columns], fontsize=12)
    ax.set_yticklabels([f"{float(i):.0%}" for i in pivot.index], fontsize=12)
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel(title, rotation=90, labelpad=15, va="bottom", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    fig.tight_layout()

    if main_title:
        fig.suptitle(main_title, fontsize=18, y=0.95)
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
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap=cmap, ax=ax, annot_kws={"fontsize": 12})
    ax.invert_yaxis()
    ax.set_xlabel("CR", fontsize=14)
    ax.set_ylabel("SELLcap", fontsize=14)
    ax.set_xticklabels([f"{float(c):.0%}" for c in pivot.columns], fontsize=12)
    ax.set_yticklabels([f"{float(i):.0%}" for i in pivot.index], fontsize=12)
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel(title, rotation=90, labelpad=15, va="bottom", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    fig.tight_layout()

    if main_title:
        fig.suptitle(main_title, fontsize=18, y=0.95)
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
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap=cmap, ax=ax, annot_kws={"fontsize": 12})
    ax.invert_yaxis()
    ax.set_xlabel("CR", fontsize=14)
    ax.set_ylabel("BOTHcap", fontsize=14)
    ax.set_xticklabels([f"{float(c):.0%}" for c in pivot.columns], fontsize=12)
    ax.set_yticklabels([f"{float(i):.0%}" for i in pivot.index], fontsize=12)
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel(title, rotation=90, labelpad=15, va="bottom", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    fig.tight_layout()

    if main_title:
        fig.suptitle(main_title, fontsize=18, y=0.95)
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
        if wc is None:
            continue
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

    sns.heatmap(pivot1, annot=True, fmt=".1f", cmap=cmap, ax=ax1, vmin=vmin, vmax=vmax, cbar=False)
    ax1.invert_yaxis()
    ax1.set_xlabel("CR")
    ax1.set_ylabel(axis)
    ax1.set_title(title1)
    ax1.set_xticklabels(pct_labels(pivot1.columns))
    ax1.set_yticklabels(pct_labels(pivot1.index))

    sns.heatmap(
        pivot2, annot=True, fmt=".1f", cmap=cmap, ax=ax2, vmin=vmin, vmax=vmax,
        cbar=True, cbar_ax=cax,
    )
    ax2.invert_yaxis()
    ax2.set_xlabel("CR")
    ax2.set_ylabel(axis)
    ax2.set_title(title2)
    ax2.set_xticklabels(pct_labels(pivot2.columns))
    ax2.set_yticklabels(pct_labels(pivot2.index))

    if suptitle:
        cax.set_ylabel(suptitle, rotation=90, labelpad=8)

    fig.tight_layout()

    if main_title:
        fig.suptitle(main_title, fontsize=14, y=0.95)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

    # tight_layout leaves a wide gap before the colorbar; pull it next to ax2.
    pos2 = ax2.get_position()
    cax.set_position([pos2.x1 + 0.01, pos2.y0, cax.get_position().width, pos2.height])

    return fig, (ax1, ax2, cax)


# Helvetica where installed (e.g. macOS); otherwise its free clone TeX Gyre
# Heros, registered here from the user font folder if present.
for _font_file in (Path.home() / ".local/share/fonts/tex-gyre-heros").glob("*.otf"):
    font_manager.fontManager.addfont(_font_file)
_available_fonts = {f.name for f in font_manager.fontManager.ttflist}
HEATMAP_FONTS = [
    f for f in ["Helvetica", "Arial", "TeX Gyre Heros"] if f in _available_fonts
] + ["DejaVu Sans"]


def plot_heatmap_grid_two_regions(
    df: pd.DataFrame,
    region1,
    region2,
    axis,
    fixed,
    value_cols,
    title1,
    title2,
    cr_range=None,
    axis_range=None,
    main_title=None,
    region_col="region",
):
    """
    Single figure gathering the two-region heatmaps (``axis`` vs CR) of several
    metrics: one row per metric, one column per region. Each row shares its own
    color scale and has its own colorbar, labelled with the metric name.

    value_cols: list of (value_col, cmap) tuples, one per row.
    The other arguments are the same as in plot_heatmap_two_regions.
    """
    wildcard_cols = ["CR", "BUYcap", "SELLcap", "BOTHcap"]

    def build_pivot(region_value, value_col):
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

    def pct_labels(values):
        return [f"{float(v):.0%}" for v in values]

    annot_size, tick_size, label_size, title_size = 10, 13, 14, 15
    cbar_tick_size, cbar_label_size = 10, 12
    title_pad = 12  # gap between the region headers and the first row [pt]

    nrows = len(value_cols)
    # The font family is fixed on each text when it is created, so everything
    # below has to be built inside this context.
    with plt.rc_context({"font.family": HEATMAP_FONTS}):
        fig, axes = plt.subplots(
            nrows, 3, figsize=(10.5, 3.2 * nrows),
            gridspec_kw={"width_ratios": [1, 1, 0.05]}, squeeze=False,
        )

        for row, (value_col, cmap) in enumerate(value_cols):
            ax1, ax2, cax = axes[row]
            pivot1 = build_pivot(region1, value_col)
            pivot2 = build_pivot(region2, value_col)

            vmin = min(pivot1.min().min(), pivot2.min().min())
            vmax = max(pivot1.max().max(), pivot2.max().max())

            annot_kws = {"fontsize": annot_size}
            sns.heatmap(
                pivot1, annot=True, fmt=".1f", cmap=cmap, ax=ax1, vmin=vmin, vmax=vmax,
                cbar=False, annot_kws=annot_kws,
            )
            sns.heatmap(
                pivot2, annot=True, fmt=".1f", cmap=cmap, ax=ax2, vmin=vmin, vmax=vmax,
                cbar=True, cbar_ax=cax, annot_kws=annot_kws,
            )

            for ax, pivot, title in [(ax1, pivot1, title1), (ax2, pivot2, title2)]:
                ax.invert_yaxis()
                ax.grid(False)
                ax.set_xlabel("CR", fontsize=label_size, fontstyle="italic")
                ax.set_ylabel(axis, fontsize=label_size, fontstyle="italic")
                ax.set_xticklabels(pct_labels(pivot.columns), fontsize=tick_size)
                ax.set_yticklabels(pct_labels(pivot.index), fontsize=tick_size, rotation=0)
                # Region names only once, as column headers above the first row.
                if row == 0:
                    ax.set_title(title, fontsize=title_size, pad=title_pad)

            cax.grid(False)
            cax.tick_params(labelsize=cbar_tick_size)
            # Unit on its own line so the label fits the shorter rows.
            cax.set_ylabel(
                value_col.replace(" [", "\n["), rotation=90, labelpad=8, fontsize=cbar_label_size
            )

        fig.tight_layout()
        if main_title:
            # Sit the main title just above the region headers.
            headers_top = title_pad + 1.2 * title_size
            y = axes[0, 0].get_position().y1 + (headers_top + 8) / 72 / fig.get_figheight()
            fig.suptitle(main_title, fontsize=16, fontstyle="italic", y=y, va="bottom")

        # tight_layout leaves a wide gap before the colorbars; pull them next to ax2.
        for ax1, ax2, cax in axes:
            pos2 = ax2.get_position()
            cax.set_position([pos2.x1 + 0.01, pos2.y0, cax.get_position().width, pos2.height])

    return fig, axes


#%%
# Build the combined metrics DataFrame once: each network (both regions) is
# loaded from disk exactly once, and every plot below (single-region and
# combined) reuses this in-memory DataFrame instead of reloading networks.

iberian_networks_folder = r"results/Iberic40_2035_industrial_clusters_all3h/all/networks"
nordic_networks_folder = r"results/Noridcs100_2035_industrial_clusters_all3h/all/networks"

config_plotting = yaml.safe_load(Path("config/plotting.default.yaml").read_text())
regions_iberian = gdp.read_file(r'resources/Iberic40_2035_industrial_clusters/all/regions_onshore_base_s_40.geojson').set_index("name")
regions_nordic = gdp.read_file(r'resources/Noridcs100_2035_industrial_clusters/all/regions_onshore_base_s_100.geojson').set_index("name")

# Matplotlib has no built-in sequential yellow colormap, so define one.
YELLOWS = mcolors.LinearSegmentedColormap.from_list(
    "Yellows", ["#fffde7", "#fff176", "#fdd835", "#f9a825"]
)

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
fig, ax = plot_heatmap_sellcap_vs_cr(df, 0, 0, "CO₂ Captured Cluster [Mtons/year]", "Purples", cr_range=None, sellcap_range=(0, 0.5), title="CO₂ Captured in Renewable Clusters [Mtons/year]", main_title="Iberian Peninsula - BUYcap=0%, BOTHcap=0%")
fig, ax = plot_heatmap_bothcap_vs_cr(df, 0, 0, "CO₂ Captured Cluster [Mtons/year]", "Purples", cr_range=None, bothcap_range=(0, 0.5), title="CO₂ Captured in Renewable Clusters [Mtons/year]", main_title="Iberian Peninsula 2035 - BUYcap=0%, SELLcap=0%")

fig, ax = plot_heatmap_buycap_vs_cr(df, 0, 0, "CO₂ Price [€/ton]", "Reds", cr_range=None, buycap_range=(0, 1), title="CO₂ Marginal Price [€/ton]", main_title="Iberian Peninsula - SELLcap=0%, BOTHcap=0%")
fig, ax = plot_heatmap_sellcap_vs_cr(df, 0, 0, "CO₂ Price [€/ton]", "Reds", cr_range=None, sellcap_range=(0, 0.5), title="CO₂ Marginal Price [€/ton]", main_title="Iberian Peninsula - BUYcap=0%, BOTHcap=0%")
fig, ax = plot_heatmap_bothcap_vs_cr(df, 0, 0, "CO₂ Price [€/ton]", "Reds", cr_range=None, bothcap_range=(0, 0.5), title="CO₂ Marginal Price [€/ton]", main_title="Iberian Peninsula 2035 - BUYcap=0%, SELLcap=0%")

fig, ax = plot_heatmap_buycap_vs_cr(df, 0, 0, "Renewable Capacity Cluster [GW]", "Greens", cr_range=None, buycap_range=(0, 1), title="Renewable Capacity in Clusters [GW]", main_title="Iberian Peninsula - SELLcap=0%, BOTHcap=0%")
fig, ax = plot_heatmap_sellcap_vs_cr(df, 0, 0, "Renewable Capacity Cluster [GW]", "Greens", cr_range=None, sellcap_range=(0, 0.5), title="Renewable Capacity in Clusters [GW]", main_title="Iberian Peninsula - BUYcap=0%, BOTHcap=0%")
fig, ax = plot_heatmap_bothcap_vs_cr(df, 0, 0, "Renewable Capacity Cluster [GW]", "Greens", cr_range=None, bothcap_range=(0, 0.5), title="Renewable Capacity in Clusters [GW]", main_title="Iberian Peninsula 2035 - BUYcap=0%, SELLcap=0%")

fig, ax = plot_heatmap_buycap_vs_cr(df, 0, 0, "Share of Curtailment in the Cluster[%]", YELLOWS, cr_range=None, buycap_range=(0, 1), title="Share of Curtailment in Renewable Clusters [%]", main_title="Iberian Peninsula - SELLcap=0%, BOTHcap=0%")
fig, ax = plot_heatmap_sellcap_vs_cr(df, 0, 0, "Share of Curtailment in the Cluster[%]", YELLOWS, cr_range=None, sellcap_range=(0, 0.5), title="Share of Curtailment in Renewable Clusters [%]", main_title="Iberian Peninsula - BUYcap=0%, BOTHcap=0%")
fig, ax = plot_heatmap_bothcap_vs_cr(df, 0, 0, "Share of Curtailment in the Cluster[%]", YELLOWS, cr_range=None, bothcap_range=(0, 0.5), title="Share of Curtailment in Renewable Clusters [%]", main_title="Iberian Peninsula 2035 - BUYcap=0%, SELLcap=0%")


# %%
#Noridc Countries

df = df_nordic

#%%
fig, ax = plot_heatmap_buycap_vs_cr(df, 0, 0, "CO₂ Captured Cluster [Mtons/year]", "Purples", cr_range=None, buycap_range=(0, 1), title="CO₂ Captured in Renewable Clusters [Mtons/year]", main_title="Nordics Countries - SELLcap=0%")
fig, ax = plot_heatmap_sellcap_vs_cr(df, 0, 0, "CO₂ Captured Cluster [Mtons/year]", "Purples", cr_range=None, sellcap_range=(0, 0.5), title="CO₂ Captured in Renewable Clusters [Mtons/year]", main_title="Nordic Countries - BUYcap=0%")
fig, ax = plot_heatmap_bothcap_vs_cr(df, 0, 0, "CO₂ Captured Cluster [Mtons/year]", "Purples", cr_range=None, bothcap_range=(0, 0.5), title="CO₂ Captured in Renewable Clusters [Mtons/year]", main_title="Nordics 2035 - BUYcap=0%, SELLcap=0%")

fig, ax = plot_heatmap_buycap_vs_cr(df, 0, 0, "CO₂ Price [€/ton]", "Reds", cr_range=None, buycap_range=(0, 1), title="CO₂ Marginal Price [€/ton]", main_title="Nordic Countries- SELLcap=0%")
fig, ax = plot_heatmap_sellcap_vs_cr(df, 0, 0, "CO₂ Price [€/ton]", "Reds", cr_range=None, sellcap_range=(0, 0.5), title="CO₂ Marginal Price [€/ton]", main_title="Nordic Countries - BUYcap=0%")
fig, ax = plot_heatmap_bothcap_vs_cr(df, 0, 0, "CO₂ Price [€/ton]", "Reds", cr_range=None, bothcap_range=(0, 0.5), title="CO₂ Marginal Price [€/ton]", main_title="Nordics 2035 - BUYcap=0%, SELLcap=0%")

fig, ax = plot_heatmap_buycap_vs_cr(df, 0, 0, "Renewable Capacity Cluster [GW]", "Greens", cr_range=None, buycap_range=(0, 1), title="Renewable Capacity in Clusters [GW]", main_title="Nordic Countries - SELLcap=0%")
fig, ax = plot_heatmap_sellcap_vs_cr(df, 0, 0, "Renewable Capacity Cluster [GW]", "Greens", cr_range=None, sellcap_range=(0, 0.5), title="Renewable Capacity in Clusters [GW]", main_title="Nordics Countries - BUYcap=0%")
fig, ax = plot_heatmap_bothcap_vs_cr(df, 0, 0, "Renewable Capacity Cluster [GW]", "Greens", cr_range=None, bothcap_range=(0, 0.5), title="Renewable Capacity in Clusters [GW]", main_title="Nordics 2035 - BUYcap=0%, SELLcap=0%")

fig, ax = plot_heatmap_buycap_vs_cr(df, 0, 0, "Share of Curtailment in the Cluster[%]", YELLOWS, cr_range=None, buycap_range=(0, 1), title="Share of Curtailment in Renewable Clusters [%]", main_title="Nordic Countries - SELLcap=0%")
fig, ax = plot_heatmap_sellcap_vs_cr(df, 0, 0, "Share of Curtailment in the Cluster[%]", YELLOWS, cr_range=None, sellcap_range=(0, 0.5), title="Share of Curtailment in Renewable Clusters [%]", main_title="Nordic Countries - BUYcap=0%")
fig, ax = plot_heatmap_bothcap_vs_cr(df, 0, 0, "Share of Curtailment in the Cluster[%]", YELLOWS, cr_range=None, bothcap_range=(0, 0.5), title="Share of Curtailment in Renewable Clusters [%]", main_title="Nordics 2035 - BUYcap=0%, SELLcap=0%")

#%%
# Combined Iberian Peninsula + Nordics heatmaps, one figure per metric, shared color scale/legend.
# Reuses df_all (built once above) - no network reloading here.

for value_col, cmap in [
    ("CO₂ Captured Cluster [Mtons/year]", "Purples"),
    ("CO₂ Price [€/ton]", "Reds"),
    ("Renewable Capacity Cluster [GW]", "Greens"),
    ("Share of Curtailment in the Cluster[%]", YELLOWS),
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
        axis_range=(0, 0.5),
        suptitle=value_col,
        main_title="BUYcap=0%",
    )

    fig, axes = plot_heatmap_two_regions(
        df_all,
        REGION_IBERIAN,
        REGION_NORDIC,
        axis="BOTHcap",
        fixed={"BUYcap": 0, "SELLcap": 0},
        value_col=value_col,
        title1=REGION_IBERIAN,
        title2=REGION_NORDIC,
        cmap=cmap,
        cr_range=None,
        axis_range=(0, 0.5),
        suptitle=value_col,
        main_title="BUYcap=0%",
    )

#%%
# Same combined heatmaps gathered in a single figure per wildcard: one row per
# metric, one column per region.

VALUE_COLS = [
    # ("CO₂ Captured Cluster [Mtons/year]", "Purples"),
    ("CO₂ Price [€/ton]", "Reds"),
    # ("Renewable Capacity Cluster [GW]", "Greens"),
    # ("Share of Curtailment in the Cluster[%]", YELLOWS),
]

PICTURES_DIR = Path("/work/users/rofrug/projects/pypsa-eur/pictures")
PICTURES_DIR.mkdir(parents=True, exist_ok=True)

for axis, fixed, axis_range in [
    ("BUYcap", {"SELLcap": 0, "BOTHcap": 0}, (0, 1)),
    ("SELLcap", {"BUYcap": 0, "BOTHcap": 0}, (0, 0.5)),
    ("BOTHcap", {"BUYcap": 0, "SELLcap": 0}, (0, 0.5)),
]:
    fig, axes = plot_heatmap_grid_two_regions(
        df_all,
        REGION_IBERIAN,
        REGION_NORDIC,
        axis=axis,
        fixed=fixed,
        value_cols=VALUE_COLS,
        title1=REGION_IBERIAN,
        title2=REGION_NORDIC,
        cr_range=None,
        axis_range=axis_range,
        main_title=", ".join(f"{k}={v:.0%}" for k, v in fixed.items()),
    )
    fig.savefig(PICTURES_DIR / f"heatmap_co2price_{axis}.pdf", bbox_inches="tight")
    plt.show()

 #%%
# Per-node capacities and full load hours of the renewable cluster components

CLUSTER_GENERATORS = ["solar", "solar-hsat", "onwind"]
CLUSTER_LINKS = ["H2 Electrolysis", "urban decentral DAC", "methanolisation", "Fischer-Tropsch"]
CLUSTER_TECHS = CLUSTER_GENERATORS + CLUSTER_LINKS


def filter_networks(df: pd.DataFrame, combinations) -> pd.DataFrame:
    """
    Keep the rows of ``df`` whose (CR, BUYcap, SELLcap, BOTHcap) matches one
    of the given ``combinations`` (an iterable of 4-tuples).
    Combinations with no matching network are reported and skipped.
    """
    wildcard_cols = ["CR", "BUYcap", "SELLcap", "BOTHcap"]
    df = df.drop_duplicates(subset=wildcard_cols, keep="first")
    combinations = np.atleast_2d(np.asarray(list(combinations), dtype=float))
    matches = np.isclose(
        df[wildcard_cols].to_numpy()[:, None, :], combinations[None, :, :]
    ).all(axis=2)
    for combo in combinations[~matches.any(axis=0)]:
        print(f"No network found for (CR, BUYcap, SELLcap, BOTHcap) = {tuple(combo.tolist())}")
    return df.loc[matches.any(axis=1)]


def cluster_generators(n, tech) -> pd.DataFrame:
    """
    The '<node> <class> <tech> renewable cluster' generators of one tech, with
    a 'node' column taken from their bus ('<node> renewable cluster').
    """
    gens = n.generators.loc[n.generators.index.str.endswith(f" {tech} renewable cluster")].copy()
    gens["node"] = gens.bus.str.removesuffix(" renewable cluster")
    return gens


def cluster_links(n, link_name) -> pd.DataFrame:
    """
    The '<node> <link_name> renewable cluster' links, with a 'node' column.
    """
    suffix = f" {link_name} renewable cluster"
    links = n.links.loc[n.links.index.str.endswith(suffix)].copy()
    links["node"] = links.index.str.removesuffix(suffix)
    return links


def cluster_nodes(n, min_p_nom_opt=1.0) -> list:
    """
    Nodes having at least one solar/solar-hsat/onwind renewable cluster
    generator with p_nom_opt > min_p_nom_opt.
    """
    nodes = set()
    for tech in CLUSTER_GENERATORS:
        gens = cluster_generators(n, tech)
        nodes.update(gens.loc[gens.p_nom_opt > min_p_nom_opt, "node"])
    return sorted(nodes)


def cluster_capacities_and_flh(n, min_p_nom_opt=1.0, top_n=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (capacities [MW electricity], full load hours [h]) of the renewable
    cluster generators/links of one network, indexed by node with one column
    per tech. Generators use p, links use p0 (capacity is defined on bus0);
    several generators of the same tech at one node are summed.
    Full load hours are computed on the native p_nom_opt.
    Components with p_nom_opt < min_p_nom_opt are treated as p_nom_opt = 0.
    Missing components (or zero capacity) give NaN full load hours.
    If top_n is given, only the top_n nodes by total generator capacity
    (solar + solar-hsat + onwind) are kept, largest first.
    """
    nodes = cluster_nodes(n, min_p_nom_opt)
    capacities = pd.DataFrame(0.0, index=nodes, columns=CLUSTER_TECHS)
    flh = pd.DataFrame(np.nan, index=nodes, columns=CLUSTER_TECHS)

    weights = n.snapshot_weightings.generators
    gen_energy = n.generators_t.p.multiply(weights, axis=0).sum()
    link_energy = n.links_t.p0.multiply(weights, axis=0).sum()

    for tech in CLUSTER_TECHS:
        if tech in CLUSTER_GENERATORS:
            comps, energy = cluster_generators(n, tech), gen_energy
        else:
            comps, energy = cluster_links(n, tech), link_energy
        comps = comps.loc[comps.node.isin(nodes)]
        comps["energy"] = energy.reindex(comps.index).fillna(0.0)

        # Capacity in MW of electricity: methanolisation and Fischer-Tropsch
        # p_nom_opt is in MW of H2 (bus0)
        comps["p_nom_el"] = comps.p_nom_opt
        if tech == "methanolisation":
            # efficiency2: MWh electricity consumed per MWh H2
            comps["p_nom_el"] *= comps.efficiency2.abs()
        elif tech == "Fischer-Tropsch":
            # efficiency of the node's electrolyser: MWh H2 per MWh electricity
            electrolysis_eff = cluster_links(n, "H2 Electrolysis").set_index("node").efficiency
            comps["p_nom_el"] /= comps.node.map(electrolysis_eff)

        # Components with p_nom_opt < min_p_nom_opt (native units) count as not built
        comps.loc[comps.p_nom_opt < min_p_nom_opt, ["p_nom_opt", "p_nom_el", "energy"]] = 0.0

        per_node = comps.groupby("node")[["p_nom_opt", "p_nom_el", "energy"]].sum()

        capacities.loc[per_node.index, tech] = per_node.p_nom_el
        built = per_node.index[per_node.p_nom_opt > 0]
        flh.loc[built, tech] = per_node.energy[built] / per_node.p_nom_opt[built]

    # Keep only the top_n nodes by total generator capacity (solar + solar-hsat + onwind),
    # largest first
    if top_n is not None:
        top_nodes = capacities[CLUSTER_GENERATORS].sum(axis=1).nlargest(top_n).index
        capacities, flh = capacities.loc[top_nodes], flh.loc[top_nodes]

    return capacities, flh


def build_cluster_dicts(df: pd.DataFrame, combinations, top_n):
    """
    For every network in ``df`` matching one of the given
    (CR, BUYcap, SELLcap, BOTHcap) ``combinations``, build the per-node
    capacity and full-load-hours DataFrames, keeping the ``top_n`` nodes
    with the largest total generator capacity (None keeps all nodes).

    Returns two dicts keyed by the (CR, BUYcap, SELLcap, BOTHcap) tuple.
    """
    renewable_cluster_capacities = {}
    renewable_cluster_full_load_hours = {}
    for row in filter_networks(df, combinations).itertuples():
        n = pypsa.Network(str(row.path))
        key = (row.CR, row.BUYcap, row.SELLcap, row.BOTHcap)
        renewable_cluster_capacities[key], renewable_cluster_full_load_hours[key] = (
            cluster_capacities_and_flh(n, top_n=top_n)
        )
    return renewable_cluster_capacities, renewable_cluster_full_load_hours


# (CR, BUYcap, SELLcap, BOTHcap)
combinations = [
    (0.4, 0, 0, 0),
    (0.4, 0.5, 0, 0),
    (0.4, 0, 0, 0.2),
]

renewable_cluster_capacities, renewable_cluster_full_load_hours = build_cluster_dicts(
    df_iberian, combinations, 5
)
renewable_cluster_capacities_nordic, renewable_cluster_full_load_hours_nordic = build_cluster_dicts(
    df_nordic, combinations, 5 
)

# Region name -> (capacities dict, full load hours dict), used by the plots below
CLUSTER_RESULTS = {
    REGION_IBERIAN: (renewable_cluster_capacities, renewable_cluster_full_load_hours),
    REGION_NORDIC: (renewable_cluster_capacities_nordic, renewable_cluster_full_load_hours_nordic),
}

#%%
# Strip plots of capacity and full load hours per tech, one colour per node


def natural_key(node: str) -> list:
    """Sort key so that 'ES0 2' comes before 'ES0 10'."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", node)]


def node_colors(nodes) -> dict:
    """
    Fixed colour per node (same node -> same colour in every figure).
    Up to 60 nodes use the tab20/tab20b/tab20c qualitative colours,
    more than that fall back to sampling the 'turbo' colormap.
    """
    nodes = sorted(nodes, key=natural_key)
    if len(nodes) <= 60:
        palette = [
            plt.get_cmap(name)(i) for name in ["tab20", "tab20b", "tab20c"] for i in range(20)
        ]
    else:
        palette = plt.get_cmap("turbo")(np.linspace(0, 1, len(nodes)))
    return dict(zip(nodes, palette))


def plot_cluster_capacities_and_flh(capacities, flh, title=None, log_capacity=False, linthresh=0.01):
    """
    One figure with two strip plots side by side: capacity [MW electricity]
    (left) and full load hours [h] (right), one column per tech in
    CLUSTER_TECHS and one dot per node. A node with zero capacity for a tech
    is drawn at 0 in the capacity plot and not drawn in the full load hours plot.

    log_capacity uses a symlog capacity axis (log above ``linthresh`` MW,
    linear below it) so that zero capacities are still visible.
    """
    nodes = sorted(capacities.index, key=natural_key)
    colors = node_colors(nodes)
    x = np.arange(len(CLUSTER_TECHS))
    # Spread the nodes horizontally inside each tech column so dots don't overlap
    offsets = np.linspace(-0.3, 0.3, len(nodes)) if len(nodes) > 1 else np.zeros(1)

    fig, (ax_cap, ax_flh) = plt.subplots(1, 2, figsize=(16, 6))

    for offset, node in zip(offsets, nodes):
        cap = capacities.loc[node, CLUSTER_TECHS].fillna(0.0)
        hours = flh.loc[node, CLUSTER_TECHS]
        built = (cap > 0).to_numpy() & hours.notna().to_numpy()
        # clip_on=False so dots on the zero line / axis edges are drawn whole
        ax_cap.scatter(x + offset, cap, s=30, color=colors[node], label=node,
                       edgecolor="white", linewidth=0.5, zorder=3, clip_on=False)
        ax_flh.scatter(x[built] + offset, hours[built], s=30, color=colors[node],
                       edgecolor="white", linewidth=0.5, zorder=3, clip_on=False)

    for ax, ylabel in [(ax_cap, "Capacity [MW$_{el}$]"), (ax_flh, "Full load hours [h]")]:
        ax.set_xticks(x)
        ax.set_xticklabels(CLUSTER_TECHS, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="0.9", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        # Light separators between tech columns
        for xi in x[:-1]:
            ax.axvline(xi + 0.5, color="0.9", linewidth=0.8, zorder=0)
    if log_capacity:
        ax_cap.set_yscale("symlog", linthresh=linthresh)
    ax_cap.set_ylim(bottom=0)
    ax_flh.set_ylim(0, 8760)

    fig.legend(
        *ax_cap.get_legend_handles_labels(),
        title="Node",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        ncol=max(1, int(np.ceil(len(nodes) / 25))),
        frameon=False,
    )
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, (ax_cap, ax_flh)


for region, (caps_dict, flh_dict) in CLUSTER_RESULTS.items():
    for key, capacities in caps_dict.items():
        CR, BUYcap, SELLcap, BOTHcap = key
        if capacities.empty:
            print(f"{region}: no renewable cluster generator with p_nom_opt > 1 for {key}, skipping plot")
            continue
        fig, axes = plot_cluster_capacities_and_flh(
            capacities,
            flh_dict[key],
            title=f"{region} - CR={CR:.0%}, BUYcap={BUYcap:.0%}, SELLcap={SELLcap:.0%}, BOTHcap={BOTHcap:.0%}",
            log_capacity=False,
        )
        plt.show()

#%%
# Box plots of capacity and full load hours per tech, with the nodes as grey dots behind


def plot_cluster_boxplots(capacities, flh, title=None, log_capacity=False, linthresh=0.01):
    """
    One figure with two box plots side by side: capacity [MW electricity]
    (left) and full load hours [h] (right), one box per tech in CLUSTER_TECHS
    computed over the nodes. Boxes show the median and the first/third
    quartiles, whiskers span the full range (min to max) of the data.
    Each node is also drawn as a light grey dot behind the boxes.

    As in plot_cluster_capacities_and_flh, a node with zero capacity for a
    tech counts as 0 in the capacity plot and is left out of the full load
    hours plot.
    """
    x = np.arange(len(CLUSTER_TECHS))
    n_nodes = len(capacities.index)
    # Spread the dots horizontally inside each tech column
    offsets = np.linspace(-0.25, 0.25, n_nodes) if n_nodes > 1 else np.zeros(n_nodes)

    caps = capacities[CLUSTER_TECHS].fillna(0.0)
    hours = flh[CLUSTER_TECHS].where(caps > 0)

    fig, (ax_cap, ax_flh) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data in [(ax_cap, caps), (ax_flh, hours)]:
        # Background dots
        for xi, tech in zip(x, CLUSTER_TECHS):
            values = data[tech].to_numpy()
            valid = ~np.isnan(values)
            ax.scatter(xi + offsets[valid], values[valid], s=15, color="0.8",
                       zorder=1, clip_on=False)

        # Boxes only for techs with at least one value
        techs = [i for i, tech in enumerate(CLUSTER_TECHS) if data[tech].notna().any()]
        if not techs:
            continue
        ax.boxplot(
            [data[CLUSTER_TECHS[i]].dropna() for i in techs],
            positions=x[techs],
            widths=0.6,
            whis=(0, 100),  # whiskers at min and max
            showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor="none", edgecolor="0.2", linewidth=1.2),
            medianprops=dict(color="tab:red", linewidth=2),
            whiskerprops=dict(color="0.2", linewidth=1.2),
            capprops=dict(color="0.2", linewidth=1.2),
            zorder=2,
        )

    for ax, ylabel in [(ax_cap, "Capacity [MW$_{el}$]"), (ax_flh, "Full load hours [h]")]:
        ax.set_xticks(x)
        ax.set_xticklabels(CLUSTER_TECHS, rotation=30, ha="right")
        ax.set_xlim(-0.6, len(CLUSTER_TECHS) - 0.4)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="0.9", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    if log_capacity:
        ax_cap.set_yscale("symlog", linthresh=linthresh)
    ax_cap.set_ylim(bottom=0)
    ax_flh.set_ylim(0, 8760)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, (ax_cap, ax_flh)


for region, (caps_dict, flh_dict) in CLUSTER_RESULTS.items():
    for key, capacities in caps_dict.items():
        CR, BUYcap, SELLcap, BOTHcap = key
        if capacities.empty:
            print(f"{region}: no renewable cluster generator with p_nom_opt > 1 for {key}, skipping plot")
            continue
        fig, axes = plot_cluster_boxplots(
            capacities,
            flh_dict[key],
            title=f"{region} - CR={CR:.0%}, BUYcap={BUYcap:.0%}, SELLcap={SELLcap:.0%}, BOTHcap={BOTHcap:.0%}",
            log_capacity=False,
        )
        plt.show()

#%%
