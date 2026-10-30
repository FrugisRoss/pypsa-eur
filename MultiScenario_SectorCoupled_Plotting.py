#%%

import copy
import pypsa
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Patch, Polygon, Wedge
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
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
    'Net Balance': "#00FF00",  # Changed to green
    'onwind': "#0281d6",
    'Fischer-Tropsch': "#4e42f5",
    'Sabatier': '#ebb028',

    # Renewable cluster variants
    'methanolisation renewable cluster': '#EB28B0',
    'H2 Electrolysis renewable cluster': "#473BF5",
    'solar-hsat renewable cluster': "#870000",
    'solar renewable cluster': '#fad13c',
    'battery charger renewable cluster': "#EF7627",
    'battery discharger renewable cluster': "#0d0f5c",
    'electricity renewable cluster': "#BEBBFA",
    'electricity renewable cluster back': "#889899",
    'H2 Store renewable cluster charge': "#59786d",
    'H2 Store renewable cluster discharge': "#2A483B",
    'onwind renewable cluster': "#8ad0ff",
    'Fischer-Tropsch renewable cluster': "#c99799",
    'Sabatier renewable cluster': '#debf12',
    'electricity renewable cluster both': "#BEBBFA",

    'DAC': "#b93af0",
    'SMR CC': "#6b3a1f",
    'biogas to gas CC': "#8b4513",
    'gas for industry CC': "#5c3317",
    'process emissions CC': "#0d0d0d",
    'solid biomass for industry CC': "#5a6b23",
    'urban central gas CHP CC': "#7a4a21",
    'urban central solid biomass CHP CC': "#4f6b2f",


    'DAC renewable cluster': "#ff8fab",

    'grid connection (sell) renewable cluster':'#a69594',
    'grid connection (buy) renewable cluster':'#858ba8'

}

# Display names for the grid-connection links in plots. The underlying
# PyPSA link names ('... electricity renewable cluster' / '... electricity
# renewable cluster back') stay as-is; only the plotted labels change.
cluster_link_display_names = {
    'electricity renewable cluster': 'grid connection (sell) renewable cluster',
    'electricity renewable cluster back': 'grid connection (buy) renewable cluster',
    'urban central DAC renewable cluster': 'DAC renewable cluster',
    'urban decentral DAC renewable cluster': 'DAC renewable cluster',
}


# Display names for the grid-connection links in plots. The underlying
# PyPSA link names ('... electricity renewable cluster' / '... electricity
# renewable cluster back') stay as-is; only the plotted labels change.
cluster_link_display_names = {
    'electricity renewable cluster': 'grid connection (sell) renewable cluster',
    'electricity renewable cluster back': 'grid connection (buy) renewable cluster',
    'electricity renewable cluster both': 'grid connection (both) renewable cluster',
    'urban central DAC renewable cluster': 'DAC renewable cluster',
    'urban decentral DAC renewable cluster': 'DAC renewable cluster',
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


def load_network(networks_folder, cr, buycap, sellcap, bothcap) -> pypsa.Network:
    """
    Load the single network in ``networks_folder`` whose CR/BUYcap/SELLcap/BOTHcap
    wildcards match the given values (a missing wildcard counts as 0, as in
    parse_wildcards). If several files match, the first in sorted order is used,
    consistent with drop_duplicates(keep="first") in the plotting functions.
    """
    target = {"CR": cr, "BUYcap": buycap, "SELLcap": sellcap, "BOTHcap": bothcap}
    matches = []
    for path in sorted(Path(networks_folder).glob("*.nc")):
        wc = parse_wildcards(path)
        if wc is None:
            continue
        if all(np.isclose(wc[k], float(v)) for k, v in target.items()):
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"No network in {networks_folder} with {target}")
    if len(matches) > 1:
        print(f"Warning: {len(matches)} networks match {target}, using {matches[0].name}")
    return pypsa.Network(str(matches[0]))

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
    generators = generators[n.generators.loc[generators, "p_nom_opt"].gt(10)]

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


def cluster_grid_links(n, tech) -> pd.DataFrame:
    """
    The '<node> <suffix>' grid connection links of one CLUSTER_GRIDS tech
    (given by its plotted name), with a 'node' column.
    """
    suffix = " " + next(s for s, name in CLUSTER_GRIDS.items() if name == tech)
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


def cluster_capacities_and_flh(n, min_p_nom_opt=1.0, top_n=None, top_n_tech=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (capacities [MW electricity], full load hours [h]) of the renewable
    cluster generators/links of one network, indexed by node with one column
    per tech. Generators use p, links use p0 (capacity is defined on bus0),
    grid connection links (CLUSTER_GRIDS) use |p0|;
    several generators of the same tech at one node are summed.
    Full load hours are computed on the native p_nom_opt.
    Components with p_nom_opt < min_p_nom_opt are treated as p_nom_opt = 0.
    Missing components (or zero capacity) give NaN full load hours.
    If top_n is given, only the top_n nodes by total capacity of the
    top_n_tech techs (a tech name or a list of them, by default
    CLUSTER_GENERATORS: solar + solar-hsat + onwind) are kept, largest first.
    """
    if top_n_tech is None:
        top_n_tech = CLUSTER_GENERATORS
    elif isinstance(top_n_tech, str):
        top_n_tech = [top_n_tech]
    unknown = set(top_n_tech) - set(CLUSTER_TECHS)
    if unknown:
        raise ValueError(f"top_n_tech {sorted(unknown)} not in CLUSTER_TECHS {CLUSTER_TECHS}")

    nodes = cluster_nodes(n, min_p_nom_opt)
    capacities = pd.DataFrame(0.0, index=nodes, columns=CLUSTER_TECHS)
    flh = pd.DataFrame(np.nan, index=nodes, columns=CLUSTER_TECHS)

    weights = n.snapshot_weightings.generators
    gen_energy = n.generators_t.p.multiply(weights, axis=0).sum()
    link_energy = n.links_t.p0.multiply(weights, axis=0).sum()
    # Grid links: |p0| so that the bidirectional 'both' link counts flows in both directions
    grid_energy = n.links_t.p0.abs().multiply(weights, axis=0).sum()

    for tech in CLUSTER_TECHS:
        if tech in CLUSTER_GENERATORS:
            comps, energy = cluster_generators(n, tech), gen_energy
        elif tech in CLUSTER_GRIDS.values():
            comps, energy = cluster_grid_links(n, tech), grid_energy
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

    # Keep only the top_n nodes by total capacity of the top_n_tech techs, largest first
    if top_n is not None:
        top_nodes = capacities[top_n_tech].sum(axis=1).nlargest(top_n).index
        capacities, flh = capacities.loc[top_nodes], flh.loc[top_nodes]

    return capacities, flh


def build_cluster_dicts(df: pd.DataFrame, combinations, top_n, top_n_tech=None):
    """
    For every network in ``df`` matching one of the given
    (CR, BUYcap, SELLcap, BOTHcap) ``combinations``, build the per-node
    capacity and full-load-hours DataFrames, keeping the ``top_n`` nodes
    with the largest total capacity of the ``top_n_tech`` techs (None keeps
    all nodes; see cluster_capacities_and_flh for ``top_n_tech``).

    Returns two dicts keyed by the (CR, BUYcap, SELLcap, BOTHcap) tuple.
    """
    renewable_cluster_capacities = {}
    renewable_cluster_full_load_hours = {}
    for row in filter_networks(df, combinations).itertuples():
        n = pypsa.Network(str(row.path))
        key = (row.CR, row.BUYcap, row.SELLcap, row.BOTHcap)
        renewable_cluster_capacities[key], renewable_cluster_full_load_hours[key] = (
            cluster_capacities_and_flh(n, top_n=top_n, top_n_tech=top_n_tech)
        )
    return renewable_cluster_capacities, renewable_cluster_full_load_hours





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


def combination_label(key) -> str:
    CR, BUYcap, SELLcap, BOTHcap = key
    return f"CR={CR:.0%}, BUYcap={BUYcap:.0%}, SELLcap={SELLcap:.0%}, BOTHcap={BOTHcap:.0%}"


def non_empty_networks(caps_dict, region=None) -> list:
    """Keys of the networks with at least one node; the empty ones are reported."""
    keys = []
    for key, capacities in caps_dict.items():
        if capacities.empty:
            print(f"{region}: no renewable cluster generator with p_nom_opt > 1 for {key}, skipping plot")
        else:
            keys.append(key)
    return keys


def setup_cluster_axes(axes, log_capacity=False, linthresh=0.01):
    """
    Common formatting of a (n_networks x 2) grid of capacity / full load hours
    axes: tech tick labels, y labels, light grid, axis ranges.
    """
    x = np.arange(len(CLUSTER_TECHS))
    for ax_cap, ax_flh in axes:
        for ax, ylabel in [(ax_cap, "Capacity [MW$_{el}$]"), (ax_flh, "Full load hours [h]")]:
            ax.set_xticks(x)
            ax.set_xticklabels([t.removesuffix(" renewable cluster") for t in CLUSTER_TECHS],
                               rotation=30, ha="right")
            # sharex hides the tick labels of all but the bottom row: show them on every row
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_xlim(-0.6, len(CLUSTER_TECHS) - 0.4)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", color="0.9", zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            # Light separators between tech columns
            for xi in x[:-1]:
                ax.axvline(xi + 0.5, color="0.9", linewidth=0.8, zorder=0)
    # Axes in a column are shared, so setting the first row sets them all
    if log_capacity:
        axes[0, 0].set_yscale("symlog", linthresh=linthresh)
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 1].set_ylim(0, 8760)


def plot_cluster_capacities_and_flh(caps_dict, flh_dict, region=None, log_capacity=False, linthresh=0.01,
                                    offgrid_key=None, min_gen_capacity=50.0, min_link_capacity=5.0):
    """
    One figure per region with one row per network (combination) and two
    strip plots per row: capacity [MW electricity] (left) and full load
    hours [h] (right), one column per tech in CLUSTER_TECHS and one dot per
    node. A node with zero capacity for a tech is not drawn in either plot.

    In each network only the nodes whose total generator capacity
    (CLUSTER_GENERATORS: solar + solar-hsat + onwind) is at least
    ``min_gen_capacity`` MW are drawn. The CLUSTER_LINKS data (capacity and
    full load hours) of a node is drawn only if the capacity [MW electricity]
    of its methanolisation or of its Fischer-Tropsch is above
    ``min_link_capacity`` MW.

    All capacity plots share the same y axis, as do all full load hours
    plots. Each node has the same colour and horizontal position in every
    row, and a single legend lists all nodes of the region.

    If offgrid_key, the (CR, BUYcap, SELLcap, BOTHcap) of the off-grid
    network, is given, the nodes of that network are drawn as stars instead
    of dots in every row.

    log_capacity uses a symlog capacity axis (log above ``linthresh`` MW,
    linear below it) so that very small capacities are still visible.
    """
    keys = non_empty_networks(caps_dict, region)
    if not keys:
        return None, None

    # Per network, the nodes with total generator capacity >= min_gen_capacity
    def large_nodes(key):
        gen_capacity = caps_dict[key][CLUSTER_GENERATORS].fillna(0.0).sum(axis=1)
        return gen_capacity.index[gen_capacity >= min_gen_capacity]

    plotted_nodes = {key: large_nodes(key) for key in caps_dict}

    # Nodes of the off-grid network (keys matched with a tolerance, they are floats)
    offgrid_nodes = set()
    if offgrid_key is not None:
        matching = [key for key in caps_dict if np.allclose(key, offgrid_key)]
        if matching:
            offgrid_nodes = set(plotted_nodes[matching[0]])
        else:
            print(f"{region}: no network found for off-grid case {offgrid_key}, no stars drawn")

    # Union of the plotted nodes over all networks of the region
    nodes = sorted(set().union(*(plotted_nodes[key] for key in keys)), key=natural_key)
    if not nodes:
        print(f"{region}: no node with generator capacity >= {min_gen_capacity} MW, skipping plot")
        return None, None
    colors = node_colors(nodes)
    x = np.arange(len(CLUSTER_TECHS))
    # Spread the nodes horizontally inside each tech column so dots don't overlap
    offsets = dict(zip(nodes, np.linspace(-0.3, 0.3, len(nodes)) if len(nodes) > 1 else [0.0]))

    fig, axes = plt.subplots(
        len(keys), 2, figsize=(16, 4.5 * len(keys)), sharex=True, sharey="col", squeeze=False
    )

    for (ax_cap, ax_flh), key in zip(axes, keys):
        capacities, flh = caps_dict[key], flh_dict[key]
        for node in plotted_nodes[key]:
            cap = capacities.loc[node, CLUSTER_TECHS].fillna(0.0)
            hours = flh.loc[node, CLUSTER_TECHS]
            # Links drawn only if methanolisation or Fischer-Tropsch capacity
            # [MW electricity] > min_link_capacity
            if not (cap[["methanolisation", "Fischer-Tropsch"]] > min_link_capacity).any():
                cap[CLUSTER_LINKS] = 0.0
            has_cap = (cap > 0).to_numpy()
            built = has_cap & hours.notna().to_numpy()
            # Stars for the nodes of the off-grid network, dots otherwise
            style = dict(marker="*", s=90) if node in offgrid_nodes else dict(marker="o", s=30)
            # clip_on=False so dots on the axis edges are drawn whole
            ax_cap.scatter(x[has_cap] + offsets[node], cap[has_cap], color=colors[node],
                           edgecolor="white", linewidth=0.5, zorder=3, clip_on=False, **style)
            ax_flh.scatter(x[built] + offsets[node], hours[built], color=colors[node],
                           edgecolor="white", linewidth=0.5, zorder=3, clip_on=False, **style)
        ax_cap.set_title(combination_label(key), loc="left", fontsize=11)

    setup_cluster_axes(axes, log_capacity, linthresh)

    # One legend entry per node of the region, whatever row it appears in
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=6, color=colors[node],
                   markeredgecolor="white", label=node)
        for node in nodes
    ]
    if offgrid_nodes:
        handles.append(
            plt.Line2D([], [], marker="*", linestyle="", markersize=10, color="0.4",
                       markeredgecolor="white", label="clusters appearing also\nin the offgrid case")
        )
    fig.legend(
        handles=handles,
        title="Node",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        ncol=max(1, int(np.ceil(len(nodes) / 25))),
        frameon=False,
    )
    if region is not None:
        fig.suptitle(region, fontsize=14)
    fig.tight_layout()
    return fig, axes


# Box plots of capacity and full load hours per tech, with the nodes as grey dots behind


def plot_cluster_boxplots(caps_dict, flh_dict, region=None, log_capacity=False, linthresh=0.01):
    """
    One figure per region with one row per network (combination) and two
    box plots per row: capacity [MW electricity] (left) and full load hours
    [h] (right), one box per tech in CLUSTER_TECHS computed over the nodes.
    Boxes show the median and the first/third quartiles, whiskers span the
    full range (min to max) of the data. Each node is also drawn as a light
    grey dot behind the boxes, at the same horizontal position in every row.

    All capacity plots share the same y axis, as do all full load hours plots.
    A node with zero capacity for a
    tech counts as 0 in the capacity plot and is left out of the full load
    hours plot.
    """
    keys = non_empty_networks(caps_dict, region)
    if not keys:
        return None, None

    nodes = sorted(set().union(*(caps_dict[key].index for key in keys)), key=natural_key)
    x = np.arange(len(CLUSTER_TECHS))
    # Spread the dots horizontally inside each tech column
    offsets = pd.Series(
        np.linspace(-0.25, 0.25, len(nodes)) if len(nodes) > 1 else [0.0], index=nodes
    )

    fig, axes = plt.subplots(
        len(keys), 2, figsize=(14, 4.5 * len(keys)), sharex=True, sharey="col", squeeze=False
    )

    for (ax_cap, ax_flh), key in zip(axes, keys):
        caps = caps_dict[key][CLUSTER_TECHS].fillna(0.0)
        hours = flh_dict[key][CLUSTER_TECHS].where(caps > 0)
        node_offsets = offsets[caps.index].to_numpy()

        for ax, data in [(ax_cap, caps), (ax_flh, hours)]:
            # Background dots
            for xi, tech in zip(x, CLUSTER_TECHS):
                values = data[tech].to_numpy()
                valid = ~np.isnan(values)
                ax.scatter(xi + node_offsets[valid], values[valid], s=15, color="0.8",
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
        ax_cap.set_title(combination_label(key), loc="left", fontsize=11)

    setup_cluster_axes(axes, log_capacity, linthresh)

    if region is not None:
        fig.suptitle(region, fontsize=14)
    fig.tight_layout()
    return fig, axes


# AC balance maps of the renewable cluster, one panel per combination

def cluster_electricity_balance(n, nodes=None, unit="TWh") -> pd.Series:
    """
    Annual electricity balance of each '<node> renewable cluster' bus,
    indexed by (bus, carrier), in ``unit``. Positive = supply, negative =
    consumption. Same bookkeeping as plot_electricity_balance_by_cluster.
    """
    weights = n.snapshot_weightings.generators

    # AC buses outside the cluster
    ac_buses = n.buses.index[
        (n.buses.carrier == "AC") & ~n.buses.index.str.endswith(" renewable cluster")
    ]
    if nodes is None:
        nodes = [b for b in ac_buses if f"{b} renewable cluster" in n.buses.index]

    balance = {}
    for node in nodes:
        bal = {}

        for gen in n.generators.index[
            n.generators.index.str.startswith(node + " ")
            & n.generators.index.str.contains("renewable cluster")
        ]:
            clean_name = f"{n.generators.at[gen, 'carrier']} renewable cluster"
            bal[clean_name] = bal.get(clean_name, 0) + (
                n.generators_t["p"][gen] * weights
            ).sum()

        for link in n.links.index[
            n.links.index.str.startswith(node + " ")
            & n.links.index.str.contains("renewable cluster")
        ]:
            clean_name = link[len(node):].lstrip()
            clean_name = cluster_link_display_names.get(clean_name, clean_name)

            if "methanolisation" in link:
                # methanolisation electricity input is on bus2
                value = -(n.links_t["p2"][link] * weights).sum()
            elif "Electrolysis" in link or "DAC" in link or "battery charger" in link:
                value = -(n.links_t["p0"][link] * weights).sum()
            elif "battery discharger" in link:
                value = -(n.links_t["p1"][link] * weights).sum()
            elif link == f"{node} electricity renewable cluster":
                value = -(n.links_t["p0"][link] * weights).sum()
            elif link == f"{node} electricity renewable cluster back":
                value = (n.links_t["p0"][link] * weights).sum()
            elif link == f"{node} electricity renewable cluster both":
                # bidirectional grid link: positive p0 -> buy, negative -> sell
                ts = n.links_t["p0"][link] * weights
                bal["grid connection (buy) renewable cluster"] = ts.clip(lower=0).sum()
                bal["grid connection (sell) renewable cluster"] = ts.clip(upper=0).sum()
                continue
            else:
                # Fischer-Tropsch, methanol output link: no electricity bus
                continue
            bal[clean_name] = bal.get(clean_name, 0) + value

        balance[node] = pd.Series(bal, dtype=float)

    conversion = {"MWh": 1, "GWh": 1e3, "TWh": 1e6}[unit]
    bus_size = pd.concat(balance, names=["bus", "carrier"]).div(conversion)
    bus_size = bus_size[bus_size.abs() > 1e-6 * bus_size.abs().max()]
    return bus_size.sort_values(ascending=False)


def ac_branch_flows(n, unit_conversion) -> pd.Series:
    """
    Annual AC line/link flows (as in plot_balance_map.py), indexed by
    (component, name), with the '-reversed' links merged into the forward ones.
    """
    flow = n.statistics.transmission(groupby=False, bus_carrier="AC").div(unit_conversion)
    if not flow.empty:
        flow_reversed_mask = flow.index.get_level_values(1).str.contains("reversed")
        flow_reversed = flow[flow_reversed_mask].rename(
            lambda x: x.replace("-reversed", "")
        )
        flow = flow[~flow_reversed_mask].subtract(flow_reversed, fill_value=0)
    return flow


def dc_forward_links(n) -> pd.Index:
    """DC links without the '-reversed' ones."""
    return n.links.index[
        (n.links.carrier == "DC") & ~n.links.index.str.endswith("-reversed")
    ]


def branch_loading_hours(n, loading_threshold=0.98) -> pd.Series:
    """
    Hours per year with loading >= ``loading_threshold`` of every AC line and
    DC link, indexed by ("Line" | "Link", name).
    """
    # AC lines: loading = p0 if > 0 else p1, over s_max_pu * s_nom_opt
    cap = n.lines.s_nom_opt * n.lines.s_max_pu
    p0 = n.lines_t.p0
    ac_loading = p0.where(p0 > 0, n.lines_t.p1).div(cap, axis=1)

    # DC lines: forward link '<name>' plus '<name>-reversed' link;
    # loading = forward p0 if >= 1 else reversed p0, over forward p_nom_opt
    dc_fwd = dc_forward_links(n)
    link_p0 = n.links_t.p0
    fwd_p0 = link_p0.reindex(columns=dc_fwd, fill_value=0)
    back_p0 = link_p0.reindex(columns=dc_fwd + "-reversed", fill_value=0).set_axis(
        dc_fwd, axis=1
    )
    dc_loading = fwd_p0.where(fwd_p0 >= 1, back_p0).div(
        n.links.p_nom_opt[dc_fwd], axis=1
    )

    # same loading -> hours step for both line types
    loading = pd.concat({"Line": ac_loading, "Link": dc_loading}, axis=1)
    return (
        loading.ge(loading_threshold)
        .mul(n.snapshot_weightings.objective, axis=0)
        .sum()
    )


def branch_mean_shadow_price(n, mu_attr="mu_upper") -> pd.Series:
    """
    Mean absolute shadow price ``mu_attr`` of the capacity constraint of every
    AC line and DC link (forward links only), indexed by ("Line" | "Link", name).
    """
    ac_mu = n.lines_t[mu_attr].reindex(columns=n.lines.index).abs().mean()
    dc_fwd = dc_forward_links(n)
    dc_mu = n.links_t[mu_attr].reindex(columns=dc_fwd).abs().mean()
    return pd.concat({"Line": ac_mu, "Link": dc_mu}).fillna(0)


def plot_cluster_electricity_balance_map_combinations(
    df,
    combinations,
    regions,
    config_plotting,
    label_to_colors,
    boundaries,
    nodes=None,
    unit="TWh",
    bus_size_factor=0.02,
    legend_bus_sizes=(10, 20),
    min_branch_width=0.3,
    branch_width_scale=2,
    loading_threshold=0.98,
    ncols=None,
    legend_gap=0.06,
    title_gap=0.02,
    branch_func=None,
    branch_label=None,
    branch_unit="h",
):
    """
    AC balance map of the renewable cluster (as plot_cluster_electricity_balance_map)
    for every (CR, BUYcap, SELLcap, BOTHcap) in ``combinations``, one panel
    per combination in a single figure.

    The first combination is the reference: its lines are coloured by the
    branch metric ``branch_func(n)`` (a Series indexed by ("Line" | "Link",
    name); by default the hours per year with loading >= ``loading_threshold``).
    In the other panels the lines are coloured by the difference of that
    metric with respect to the reference (same diverging scale for all of them).
    Semicircle and branch widths use the same scale in every panel.
    ``legend_gap`` is the space between the bottom-left colorbar and the
    supply/consumption legends (fraction of the panel height), ``title_gap``
    the space between the panel titles and the figure title (fraction of
    the figure height).
    """
    SEMICIRCLE_CORRECTION_FACTOR = (
        2 if parse(pypsa.__version__) <= Version("0.33.2")
        else 1
    )
    settings = config_plotting["plotting"]["balance_map"]["AC"]
    # scale the config width so the map lines and the legend stay consistent
    branch_width_factor = settings["branch_factor"] * branch_width_scale
    flow_size_factor = settings["flow_factor"]

    if branch_func is None:
        branch_func = lambda n: branch_loading_hours(n, loading_threshold)
        branch_label = f"Hours with line loading ≥ {loading_threshold:.0%}"
        branch_unit = "h"

    # Networks in the order of `combinations`, the reference first
    rows = filter_networks(df, combinations)
    paths = {(r.CR, r.BUYcap, r.SELLcap, r.BOTHcap): r.path for r in rows.itertuples()}
    keys = []
    for combo in combinations:
        match = [k for k in paths if np.allclose(k, combo)]
        if match:
            keys.append(match[0])
    if not keys or not np.allclose(keys[0], combinations[0]):
        raise ValueError(f"No network found for the reference combination {combinations[0]}")

    # --------------------------------------------------
    # Per-network data
    # --------------------------------------------------
    networks, bus_sizes, branch_values = {}, {}, {}
    for key in keys:
        n = pypsa.Network(str(paths[key]))
        networks[key] = n
        bus_sizes[key] = cluster_electricity_balance(n, nodes=nodes, unit=unit)
        branch_values[key] = branch_func(n)

    ref_key = keys[0]
    branch_diff = {
        key: branch_values[key].sub(
            branch_values[ref_key].reindex(branch_values[key].index), fill_value=0
        )
        for key in keys[1:]
    }

    # Reference colormap: branch metric
    line_cmap = mcolors.LinearSegmentedColormap.from_list(
        "red_scale",
        ["#ffe5e5", "#b30000", "#4d0000"],
    )
    line_norm = plt.Normalize(vmin=0, vmax=branch_values[ref_key].max() or 1)

    # Difference colormap: blue = lower, red = higher metric than the reference
    diff_cmap = mcolors.LinearSegmentedColormap.from_list(
        "diff_scale",
        ["#08306b", "#2171b5", "#d9d9d9", "#cb181d", "#67000d"],
    )
    diff_max = max([d.abs().max() for d in branch_diff.values()] + [0]) or 1
    diff_norm = plt.Normalize(vmin=-diff_max, vmax=diff_max)

    # --------------------------------------------------
    # Colours (legend labels without the ' renewable cluster' suffix)
    # --------------------------------------------------
    carrier_colors = pd.concat(
        [networks[key].carriers.color for key in keys]
    ).replace("", "grey")
    carrier_colors = carrier_colors[~carrier_colors.index.duplicated()]

    def get_color(c):
        return label_to_colors.get(c, carrier_colors.get(c, "lightgrey"))

    def get_label(c):
        return c.replace(" renewable cluster", "")

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------
    if ncols is None:
        ncols = len(keys)
    nrows = int(np.ceil(len(keys) / ncols))
    crs = load_projection(copy.deepcopy(config_plotting["plotting"]))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 6.5 * nrows),
        subplot_kw={"projection": crs},
        layout="constrained",
        squeeze=False,
    )
    axes = axes.ravel()
    for ax in axes[len(keys):]:
        ax.set_visible(False)

    dc_dash = (0, (1, 1.5))
    colorbars = []

    for ax, key in zip(axes, keys):
        n = networks[key]
        bus_size = bus_sizes[key]
        is_ref = key == ref_key

        colors = (
            bus_size.index.get_level_values("carrier")
            .unique()
            .to_series()
            .map(get_color)
        )

        flow = ac_branch_flows(n, settings["unit_conversion"])
        fallback = pd.Series(dtype=float)
        line_widths = flow.get("Line", fallback).abs()
        link_widths = flow.get("Link", fallback).abs()
        line_flow = flow.get("Line")
        link_flow = flow.get("Link")
        transformer_flow = flow.get("Transformer")

        # Line colours: metric (reference) or difference of metric (other combinations)
        if is_ref:
            values, cmap, norm = branch_values[key], line_cmap, line_norm
        else:
            values, cmap, norm = branch_diff[key], diff_cmap, diff_norm

        def value_to_color(v):
            return mcolors.to_hex(cmap(norm(v)))

        branch_colors = values.map(value_to_color)
        line_colors = branch_colors["Line"].reindex(n.lines.index, fill_value=value_to_color(0))

        # links on the map are indexed by the forward name (reversed merged in);
        # non-DC links keep the default pypsa link colour
        dc_fwd = dc_forward_links(n)
        link_colors = pd.Series("darkseagreen", index=link_widths.index, dtype=object)
        dc_on_map = dc_fwd.intersection(link_colors.index)
        link_colors[dc_on_map] = branch_colors["Link"][dc_on_map]

        collections = n.plot(
            ax=ax,
            bus_size=bus_size * bus_size_factor,
            bus_color=colors,
            bus_split_circle=True,
            # floor the widths: PDF viewers draw hairlines at >= 1 px, which is
            # why the thin AC lines are visible in the pypsa-eur PDF maps
            line_width=(line_widths * branch_width_factor).clip(lower=min_branch_width),
            line_color=line_colors,
            link_width=(link_widths * branch_width_factor).clip(lower=min_branch_width),
            link_color=link_colors,
            line_flow=line_flow * flow_size_factor if line_flow is not None else None,
            link_flow=link_flow * flow_size_factor if link_flow is not None else None,
            transformer_flow=transformer_flow * flow_size_factor
            if transformer_flow is not None
            else None,
            margin=0.2,
            geomap=True,
            geomap_color={"border": "darkgrey", "coastline": "darkgrey"},
            boundaries=boundaries,
        )

        # region borders between the nodes, below the network
        regions.to_crs(crs.proj4_init).plot(
            ax=ax,
            facecolor="none",
            edgecolor="lightgrey",
            linewidth=0.3,
            zorder=1,
        )

        # DC links dotted: the link segments are in the order of link_widths.index
        # (link_width, link_color and link_flow all share that index), so the
        # flow arrows drawn by n.plot are kept
        link_coll = collections["branches"].get("Link")
        if link_coll is not None and len(link_coll.get_segments()) == len(link_widths):
            link_coll.set_linestyle(
                [dc_dash if l in dc_on_map else "solid" for l in link_widths.index]
            )
            link_coll.set_capstyle("round")

        title = combination_label(key)
        ax.set_title(f"{title} (reference)" if is_ref else title, fontsize=10)

        if is_ref:
            cbar_label = f"{branch_label} [{branch_unit}]"
        else:
            cbar_label = f"Δ {branch_label[0].lower() + branch_label[1:]} vs. reference [{branch_unit}]"
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbr = fig.colorbar(
            sm,
            ax=ax,
            label=cbar_label,
            shrink=0.95,
            pad=0.03,
            aspect=50,
            orientation="horizontal",
        )
        cbr.outline.set_edgecolor("None")
        colorbars.append(cbr)

    # Freeze the constrained layout, so that the figure title and the
    # supply/consumption legends can be placed relative to the final
    # positions of the panels and colorbars
    fig.canvas.draw()
    fig.set_layout_engine("none")
    renderer = fig.canvas.get_renderer()

    # --------------------------------------------------
    # Legends (carriers of all combinations)
    # --------------------------------------------------
    legend_kwargs = {
        "loc": "upper left",
        "frameon": False,
        "alignment": "left",
        "title_fontproperties": {"weight": "bold"},
    }

    all_sizes = pd.concat(bus_sizes)
    all_carriers = all_sizes.index.get_level_values("carrier")
    pos_carriers = all_carriers[all_sizes.values > 0].unique()
    neg_carriers = all_carriers[all_sizes.values < 0].unique()
    common_carriers = pos_carriers.intersection(neg_carriers)

    def get_total_abs(c, sign):
        vals = all_sizes[all_carriers == c]
        return vals[vals * sign > 0].abs().sum()

    supp_carriers = sorted(
        set(pos_carriers) - set(common_carriers)
        | {c for c in common_carriers if get_total_abs(c, 1) >= get_total_abs(c, -1)}
    )
    cons_carriers = sorted(
        set(neg_carriers) - set(common_carriers)
        | {c for c in common_carriers if get_total_abs(c, 1) < get_total_abs(c, -1)}
    )

    # supply / consumption below the colorbar of the bottom-left panel
    ax_bottom = axes[(nrows - 1) * ncols]
    cbar_bottom = colorbars[(nrows - 1) * ncols].ax.get_tightbbox(renderer)
    legend_y = (
        ax_bottom.transAxes.inverted().transform((0, cbar_bottom.y0))[1] - legend_gap
    )
    add_legend_patches(
        ax_bottom,
        [get_color(c) for c in supp_carriers],
        [get_label(c) for c in supp_carriers],
        legend_kw={
            "bbox_to_anchor": (0, legend_y),
            "ncol": 1,
            "title": "Supply",
            **legend_kwargs,
        },
    )
    add_legend_patches(
        ax_bottom,
        [get_color(c) for c in cons_carriers],
        [get_label(c) for c in cons_carriers],
        legend_kw={
            "bbox_to_anchor": (0.5, legend_y),
            "ncol": 1,
            "title": "Consumption",
            **legend_kwargs,
        },
    )

    # sizes and line types above the top-left panel
    ax_top = axes[0]
    add_legend_semicircles(
        ax_top,
        [s * bus_size_factor * SEMICIRCLE_CORRECTION_FACTOR for s in legend_bus_sizes],
        [f"{s:g} {unit}" for s in legend_bus_sizes],
        patch_kw={"color": "#666"},
        legend_kw={"bbox_to_anchor": (0, 1), **legend_kwargs},
    )

    legend_branch_sizes = settings["branch_sizes"]
    if legend_branch_sizes is not None:
        add_legend_lines(
            ax_top,
            [s * branch_width_factor for s in legend_branch_sizes],
            [f"{s} {settings['unit']}" for s in legend_branch_sizes],
            patch_kw={"color": "#666"},
            legend_kw={"bbox_to_anchor": (0.25, 1), **legend_kwargs},
        )

    line_type_legend = ax_top.legend(
        handles=[
            Line2D([], [], color="#666", linewidth=2, label="AC line"),
            Line2D(
                [], [], color="#666", linewidth=2, linestyle=dc_dash,
                dash_capstyle="round", label="DC line",
            ),
        ],
        bbox_to_anchor=(0.55, 1),
        **legend_kwargs,
    )
    ax_top.add_artist(line_type_legend)

    # figure title just above the panel titles
    if "region" in df.columns:
        title_top = max(
            ax.title.get_window_extent(renderer).y1 for ax in axes[:len(keys)]
        )
        title_y = fig.transFigure.inverted().transform((0, title_top))[1] + title_gap
        fig.suptitle(df["region"].iloc[0], fontsize=14, y=title_y, va="bottom")

    return fig, axes, bus_sizes


def demand_weighted_price(n, demand_bus_carriers=("low voltage",)) -> pd.Series:
    """
    Demand-weighted electricity price of every AC node (renewable cluster
    buses excluded), sum_t(price(t) * demand(t)) / sum_t(demand(t)) over the
    year, in €/MWh. The price is the marginal price of the AC bus, the demand
    the sum of the loads on the buses of ``demand_bus_carriers`` at the same
    location.
    """
    weights = n.snapshot_weightings.generators

    ac_buses = n.buses.index[
        (n.buses.carrier == "AC") & ~n.buses.index.str.endswith(" renewable cluster")
    ]
    price = n.buses_t.marginal_price[ac_buses]

    # electricity loads summed per location (= AC node)
    loads = n.loads.index[n.loads.bus.map(n.buses.carrier).isin(demand_bus_carriers)]
    demand = (
        n.loads_t.p[loads]
        .T.groupby(n.loads.bus[loads].map(n.buses.location))
        .sum()
        .T.reindex(columns=ac_buses, fill_value=0)
    )

    weighted_demand = demand.mul(weights, axis=0)
    return (price * weighted_demand).sum() / weighted_demand.sum()


def ac_marginal_prices(n) -> pd.DataFrame:
    """
    Marginal electricity price time series of every AC node (renewable cluster
    buses excluded), in €/MWh.
    """
    ac_buses = n.buses.index[
        (n.buses.carrier == "AC") & ~n.buses.index.str.endswith(" renewable cluster")
    ]
    return n.buses_t.marginal_price[ac_buses]


def max_price(n) -> pd.Series:
    """Maximum electricity price of every AC node over the year, in €/MWh."""
    return ac_marginal_prices(n).max()


def min_price(n) -> pd.Series:
    """Minimum electricity price of every AC node over the year, in €/MWh."""
    return ac_marginal_prices(n).min()


def plot_cluster_electricity_balance_price_map_combinations(
    df,
    combinations,
    regions,
    config_plotting,
    label_to_colors,
    boundaries,
    nodes=None,
    unit="TWh",
    bus_size_factor=0.02,
    legend_bus_sizes=(10, 20),
    min_branch_width=0.3,
    branch_width_scale=2,
    demand_bus_carriers=("low voltage",),
    price_cmap="Greens",
    ncols=None,
    legend_gap=0.06,
    title_gap=0.02,
    price_func=None,
    price_label="Demand-weighted electricity price",
    color_mode="difference",
    show_branches=True,
):
    """
    Same maps as plot_cluster_electricity_balance_map_combinations, but the
    colormap is on the nodes (onshore regions) instead of on the lines.

    The regions are coloured by the electricity price metric ``price_func(n)``
    (a Series per AC node; by default the demand-weighted price, see
    demand_weighted_price), depending on ``color_mode``:

    - "difference": the first combination is the reference, coloured by the
      price; the other panels are coloured by the difference of that price
      with respect to the reference (red = higher, blue = lower, same
      diverging scale for all of them).
    - "absolute": every panel coloured by its own price on the same scale,
      with a single colorbar on the right of the figure.
    - "none": regions not coloured, only the maps.

    AC lines and DC links keep the AC / DC colours of ``config_plotting``;
    with ``show_branches=False`` they are not drawn (nor their legends).
    """
    if color_mode not in ("difference", "absolute", "none"):
        raise ValueError(f"Unknown color_mode {color_mode!r}")
    is_diff = color_mode == "difference"

    if price_func is None:
        price_func = lambda n: demand_weighted_price(n, demand_bus_carriers)

    SEMICIRCLE_CORRECTION_FACTOR = (
        2 if parse(pypsa.__version__) <= Version("0.33.2")
        else 1
    )
    settings = config_plotting["plotting"]["balance_map"]["AC"]
    tech_colors = config_plotting["plotting"]["tech_colors"]
    ac_color = tech_colors.get("AC", "rosybrown")
    dc_color = tech_colors.get("DC", "darkseagreen")
    # scale the config width so the map lines and the legend stay consistent
    branch_width_factor = settings["branch_factor"] * branch_width_scale
    flow_size_factor = settings["flow_factor"]
    price_unit = settings["region_unit"]

    # Networks in the order of `combinations`, the reference first
    rows = filter_networks(df, combinations)
    paths = {(r.CR, r.BUYcap, r.SELLcap, r.BOTHcap): r.path for r in rows.itertuples()}
    keys = []
    for combo in combinations:
        match = [k for k in paths if np.allclose(k, combo)]
        if match:
            keys.append(match[0])
    if not keys:
        raise ValueError(f"No network found for the combinations {combinations}")
    if is_diff and not np.allclose(keys[0], combinations[0]):
        raise ValueError(f"No network found for the reference combination {combinations[0]}")

    # --------------------------------------------------
    # Per-network data
    # --------------------------------------------------
    networks, bus_sizes, prices = {}, {}, {}
    for key in keys:
        n = pypsa.Network(str(paths[key]))
        networks[key] = n
        bus_sizes[key] = cluster_electricity_balance(n, nodes=nodes, unit=unit)
        prices[key] = price_func(n)

    ref_key = keys[0]
    prices_diff = {
        key: prices[key].sub(prices[ref_key].reindex(prices[key].index), fill_value=0)
        for key in keys[1:]
    }

    # Price colormap: reference price ("difference") or all prices ("absolute")
    # of the regions on the map
    shown_keys = [ref_key] if is_diff else keys
    shown_price = pd.concat(
        [prices[key].reindex(regions.index) for key in shown_keys]
    ).dropna()
    price_norm = plt.Normalize(vmin=shown_price.min(), vmax=shown_price.max())
    price_cmap = plt.get_cmap(price_cmap)

    # Difference colormap: blue = lower, red = higher price than the reference
    diff_cmap = mcolors.LinearSegmentedColormap.from_list(
        "diff_scale",
        ["#08306b", "#2171b5", "#f7f7f7", "#cb181d", "#67000d"],
    )
    diff_max = max(
        [d.reindex(regions.index).abs().max() for d in prices_diff.values()] + [1e-3]
    )
    diff_norm = plt.Normalize(vmin=-diff_max, vmax=diff_max)

    # --------------------------------------------------
    # Colours (legend labels without the ' renewable cluster' suffix)
    # --------------------------------------------------
    carrier_colors = pd.concat(
        [networks[key].carriers.color for key in keys]
    ).replace("", "grey")
    carrier_colors = carrier_colors[~carrier_colors.index.duplicated()]

    def get_color(c):
        return label_to_colors.get(c, carrier_colors.get(c, "lightgrey"))

    def get_label(c):
        return c.replace(" renewable cluster", "")

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------
    if ncols is None:
        ncols = len(keys)
    nrows = int(np.ceil(len(keys) / ncols))
    crs = load_projection(copy.deepcopy(config_plotting["plotting"]))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 6.5 * nrows),
        subplot_kw={"projection": crs},
        layout="constrained",
        squeeze=False,
    )
    axes = axes.ravel()
    for ax in axes[len(keys):]:
        ax.set_visible(False)

    regions_crs = regions.to_crs(crs.proj4_init)
    colorbars = []

    for ax, key in zip(axes, keys):
        n = networks[key]
        bus_size = bus_sizes[key]
        is_ref = is_diff and key == ref_key

        colors = (
            bus_size.index.get_level_values("carrier")
            .unique()
            .to_series()
            .map(get_color)
        )

        flow = ac_branch_flows(n, settings["unit_conversion"])
        fallback = pd.Series(dtype=float)
        line_widths = flow.get("Line", fallback).abs()
        link_widths = flow.get("Link", fallback).abs()
        line_flow = flow.get("Line")
        link_flow = flow.get("Link")
        transformer_flow = flow.get("Transformer")

        # Line colours: AC or DC; non-DC links keep the default pypsa link colour
        dc_fwd = dc_forward_links(n)
        link_colors = pd.Series("darkseagreen", index=link_widths.index, dtype=object)
        link_colors[dc_fwd.intersection(link_colors.index)] = dc_color

        if not show_branches:
            line_flow = link_flow = transformer_flow = None

        collections = n.plot(
            ax=ax,
            bus_size=bus_size * bus_size_factor,
            bus_color=colors,
            bus_split_circle=True,
            # floor the widths: PDF viewers draw hairlines at >= 1 px, which is
            # why the thin AC lines are visible in the pypsa-eur PDF maps
            line_width=(line_widths * branch_width_factor).clip(lower=min_branch_width),
            line_color=ac_color,
            link_width=(link_widths * branch_width_factor).clip(lower=min_branch_width),
            link_color=link_colors,
            line_flow=line_flow * flow_size_factor if line_flow is not None else None,
            link_flow=link_flow * flow_size_factor if link_flow is not None else None,
            transformer_flow=transformer_flow * flow_size_factor
            if transformer_flow is not None
            else None,
            margin=0.2,
            geomap=True,
            geomap_color={"border": "darkgrey", "coastline": "darkgrey"},
            boundaries=boundaries,
        )

        # remove the lines/links drawn by n.plot (removing them keeps no
        # hairline in the PDF, unlike a zero width)
        if not show_branches:
            for coll in collections["branches"].values():
                coll.remove()

        title = combination_label(key)
        ax.set_title(f"{title} (reference)" if is_ref else title, fontsize=10)

        if color_mode == "none":
            # region borders between the nodes, below the network
            regions_crs.plot(
                ax=ax,
                facecolor="none",
                edgecolor="lightgrey",
                linewidth=0.3,
                zorder=1,
            )
            continue

        # Region colours: price (reference / "absolute") or difference of
        # price (other combinations of "difference")
        if is_ref or not is_diff:
            values, cmap, norm = prices[key], price_cmap, price_norm
        else:
            values, cmap, norm = prices_diff[key], diff_cmap, diff_norm

        region_values = values.reindex(regions_crs.index)
        region_colors = [
            "lightgrey" if np.isnan(v) else mcolors.to_hex(cmap(norm(v)))
            for v in region_values
        ]
        # regions below the network, with the borders between the nodes
        regions_crs.plot(
            ax=ax,
            color=region_colors,
            edgecolor="lightgrey",
            linewidth=0.3,
            zorder=1,
        )

        # one colorbar per panel only when comparing with the reference
        if not is_diff:
            continue
        if is_ref:
            cbar_label = f"{price_label} [{price_unit}]"
        else:
            cbar_label = f"Δ {price_label[0].lower() + price_label[1:]} vs. reference [{price_unit}]"
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbr = fig.colorbar(
            sm,
            ax=ax,
            label=cbar_label,
            shrink=0.95,
            pad=0.03,
            aspect=50,
            orientation="horizontal",
        )
        cbr.outline.set_edgecolor("None")
        colorbars.append(cbr)

    # "absolute": a single colorbar on the right of all the panels
    if color_mode == "absolute":
        sm = plt.cm.ScalarMappable(cmap=price_cmap, norm=price_norm)
        cbr = fig.colorbar(
            sm,
            ax=list(axes[:len(keys)]),
            label=f"{price_label} [{price_unit}]",
            location="right",
            shrink=0.8,
            pad=0.02,
            aspect=40,
        )
        cbr.outline.set_edgecolor("None")

    # Freeze the constrained layout, so that the figure title and the
    # supply/consumption legends can be placed relative to the final
    # positions of the panels and colorbars
    fig.canvas.draw()
    fig.set_layout_engine("none")
    renderer = fig.canvas.get_renderer()

    # --------------------------------------------------
    # Legends (carriers of all combinations)
    # --------------------------------------------------
    legend_kwargs = {
        "loc": "upper left",
        "frameon": False,
        "alignment": "left",
        "title_fontproperties": {"weight": "bold"},
    }

    all_sizes = pd.concat(bus_sizes)
    all_carriers = all_sizes.index.get_level_values("carrier")
    pos_carriers = all_carriers[all_sizes.values > 0].unique()
    neg_carriers = all_carriers[all_sizes.values < 0].unique()
    common_carriers = pos_carriers.intersection(neg_carriers)

    def get_total_abs(c, sign):
        vals = all_sizes[all_carriers == c]
        return vals[vals * sign > 0].abs().sum()

    supp_carriers = sorted(
        set(pos_carriers) - set(common_carriers)
        | {c for c in common_carriers if get_total_abs(c, 1) >= get_total_abs(c, -1)}
    )
    cons_carriers = sorted(
        set(neg_carriers) - set(common_carriers)
        | {c for c in common_carriers if get_total_abs(c, 1) < get_total_abs(c, -1)}
    )

    # supply / consumption below the bottom-left panel (and its colorbar)
    ax_bottom = axes[(nrows - 1) * ncols]
    if colorbars:
        bottom = colorbars[(nrows - 1) * ncols].ax.get_tightbbox(renderer).y0
    else:
        bottom = ax_bottom.get_window_extent(renderer).y0
    legend_y = (
        ax_bottom.transAxes.inverted().transform((0, bottom))[1] - legend_gap
    )
    add_legend_patches(
        ax_bottom,
        [get_color(c) for c in supp_carriers],
        [get_label(c) for c in supp_carriers],
        legend_kw={
            "bbox_to_anchor": (0, legend_y),
            "ncol": 1,
            "title": "Supply",
            **legend_kwargs,
        },
    )
    add_legend_patches(
        ax_bottom,
        [get_color(c) for c in cons_carriers],
        [get_label(c) for c in cons_carriers],
        legend_kw={
            "bbox_to_anchor": (0.5, legend_y),
            "ncol": 1,
            "title": "Consumption",
            **legend_kwargs,
        },
    )

    # sizes and line types above the top-left panel
    ax_top = axes[0]
    add_legend_semicircles(
        ax_top,
        [s * bus_size_factor * SEMICIRCLE_CORRECTION_FACTOR for s in legend_bus_sizes],
        [f"{s:g} {unit}" for s in legend_bus_sizes],
        patch_kw={"color": "#666"},
        legend_kw={"bbox_to_anchor": (0, 1), **legend_kwargs},
    )

    legend_branch_sizes = settings["branch_sizes"]
    if show_branches and legend_branch_sizes is not None:
        add_legend_lines(
            ax_top,
            [s * branch_width_factor for s in legend_branch_sizes],
            [f"{s} {settings['unit']}" for s in legend_branch_sizes],
            patch_kw={"color": "#666"},
            legend_kw={"bbox_to_anchor": (0.25, 1), **legend_kwargs},
        )

    if show_branches:
        line_type_legend = ax_top.legend(
            handles=[
                Line2D([], [], color=ac_color, linewidth=2, label="AC line"),
                Line2D([], [], color=dc_color, linewidth=2, label="DC line"),
            ],
            bbox_to_anchor=(0.55, 1),
            **legend_kwargs,
        )
        ax_top.add_artist(line_type_legend)

    # figure title just above the panel titles
    if "region" in df.columns:
        title_top = max(
            ax.title.get_window_extent(renderer).y1 for ax in axes[:len(keys)]
        )
        title_y = fig.transFigure.inverted().transform((0, title_top))[1] + title_gap
        fig.suptitle(df["region"].iloc[0], fontsize=14, y=title_y, va="bottom")

    return fig, axes, bus_sizes, prices


def plot_cluster_electricity_balance_max_price_map_combinations(*args, **kwargs):
    """
    Same as plot_cluster_electricity_balance_price_map_combinations, with the
    regions coloured by the maximum electricity price of the year (reference)
    and its difference vs. the reference (other combinations).
    """
    return plot_cluster_electricity_balance_price_map_combinations(
        *args,
        price_func=max_price,
        price_label="Maximum electricity price",
        **kwargs,
    )


def plot_cluster_electricity_balance_min_price_map_combinations(*args, **kwargs):
    """
    Same as plot_cluster_electricity_balance_price_map_combinations, with the
    regions coloured by the minimum electricity price of the year (reference)
    and its difference vs. the reference (other combinations).
    """
    return plot_cluster_electricity_balance_price_map_combinations(
        *args,
        price_func=min_price,
        price_label="Minimum electricity price",
        **kwargs,
    )


class HandlerSemicircle(HandlerBase):
    """
    Legend handler drawing an upper semicircle whose radius is the radius
    of the handle (a Wedge) in points, i.e. at its true size on the page.
    """

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        center = (width / 2 - xdescent, -ydescent)
        patch = Wedge(center, orig_handle.r, 0, 180)
        self.update_prop(patch, orig_handle, legend)
        patch.set_transform(trans)
        return [patch]


def get_projected_extent(crs, boundaries):
    """
    Bounding box [x_min, x_max, y_min, y_max], in the coordinates of the
    projection `crs`, of the lon/lat box [lon_min, lon_max, lat_min, lat_max].
    """
    lon = np.linspace(boundaries[0], boundaries[1], 100)
    lat = np.linspace(boundaries[2], boundaries[3], 100)
    lons = np.concatenate([lon, lon, np.full(100, boundaries[0]), np.full(100, boundaries[1])])
    lats = np.concatenate([np.full(100, boundaries[2]), np.full(100, boundaries[3]), lat, lat])
    points = crs.transform_points(ccrs.PlateCarree(), lons, lats)
    return [points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max()]


def pad_extent_to_aspect(extent, aspect):
    """
    Enlarge a projected extent symmetrically around its centre, in one
    direction only, until height / width equals `aspect`.
    """
    x0, x1, y0, y1 = extent
    width, height = x1 - x0, y1 - y0
    if height / width < aspect:
        pad = (width * aspect - height) / 2
        y0, y1 = y0 - pad, y1 + pad
    else:
        pad = (height / aspect - width) / 2
        x0, x1 = x0 - pad, x1 + pad
    return [x0, x1, y0, y1]


def italic_combination_label(key) -> str:
    """
    combination_label with the wildcard names in italics (as the heatmap
    axis labels) and the values upright, using mathtext.
    """
    names = ["CR", "BUYcap", "SELLcap", "BOTHcap"]
    return ", ".join(rf"$\mathit{{{name}}}$={value:.0%}" for name, value in zip(names, key))


# The font family is fixed on each text when it is created, so the whole
# figure has to be built inside this context; the mathtext italics (wildcard
# names in the titles) use the italic of the same font.
@mpl.rc_context({
    "font.family": HEATMAP_FONTS,
    "mathtext.fontset": "custom",
    "mathtext.rm": HEATMAP_FONTS[0],
    "mathtext.it": f"{HEATMAP_FONTS[0]}:italic",
    # unused, set only to avoid the missing 'cursive' font warning
    "mathtext.cal": HEATMAP_FONTS[0],
})
def plot_cluster_electricity_balance_maps_two_regions(
    region_cases,
    config_plotting,
    label_to_colors,
    nodes=None,
    unit="TWh",
    circle_scale=0.009,
    legend_sizes=(1, 5),
    fig_width=10.5,
    max_fig_height=13.6,
    supply_ncol=1,
    consumption_ncol=1,
    save_path=None,
):
    """
    Electricity balance of the renewable clusters (as
    plot_cluster_electricity_balance_price_map_combinations with
    color_mode="none" and show_branches=False) for several combinations of
    several regions in one figure: one column per region (in the order of
    ``region_cases``, e.g. Iberian Peninsula on the left and Nordic Countries
    on the right), one row per combination.

    Same layout and fonts as plot_co2_stored_balance_maps of
    MultiScenario_BaseSectorCoupled_Plotting.py: all the maps have the same
    size on the page (the boundaries of each region are enlarged, in one
    direction only, to a common height/width ratio) and the circles are
    sized relative to the map width, so the same energy gives the same
    circle on every map. This allows a single circle-size legend, placed
    below the maps to the left of the supply and consumption legends.

    Parameters
    ----------
    region_cases : list of dict
        One dict per region (column) with keys 'df' (metrics DataFrame with
        the network paths), 'combinations' (list of (CR, BUYcap, SELLcap,
        BOTHcap)), 'regions' (GeoDataFrame indexed by node name),
        'boundaries' ([lon_min, lon_max, lat_min, lat_max]) and 'title'.
    circle_scale : float
        Radius of a 1 ``unit`` semicircle as a fraction of the map width
        (radius = sqrt(size) * circle_scale * map width).
    legend_sizes : list of float
        Sizes shown in the circle-size legend, in ``unit``.
    fig_width : float
        Maximum figure width in inches; the figure is then cropped to the
        maps (no empty margins on the sides) and the height follows from the
        maps. The default is the width of plot_co2_stored_balance_maps: both
        figures included at the same scale (e.g. ``scale=0.6`` in LaTeX,
        10.5 in -> 6.3 in, the text width of an A4 page) show the text at
        the same size.
    max_fig_height : float
        Maximum figure height in inches. The default, scaled like the width
        (10.5 in -> 6.3 in), gives ~8.2 in on the page: the text height of an
        A4 page with 2.5 cm margins, leaving room for a caption of about
        5 lines. If the maps would make the figure taller, they are shrunk
        (and centred horizontally) until it fits.
    supply_ncol, consumption_ncol : int
        Number of columns of the supply and of the consumption legend.
    save_path : str or Path, optional
        If given, the figure is saved there.

    Returns
    -------
    bus_sizes : list (one per region) of dict {combination: balance Series}.
    """
    legend_sizes = sorted(legend_sizes, reverse=True)

    # --------------------------------------------------
    # Data: networks of every region in the order of its combinations
    # --------------------------------------------------
    panels = []  # per region: list of (combination, network, balance)
    carrier_colors = {}
    for case in region_cases:
        rows = filter_networks(case["df"], case["combinations"])
        paths = {(r.CR, r.BUYcap, r.SELLcap, r.BOTHcap): r.path for r in rows.itertuples()}
        region_panels = []
        for combo in case["combinations"]:
            match = [k for k in paths if np.allclose(k, combo)]
            if not match:
                continue
            n = pypsa.Network(str(paths[match[0]]))
            bus_size = cluster_electricity_balance(n, nodes=nodes, unit=unit)
            region_panels.append((match[0], n, bus_size))

            for c in bus_size.index.unique("carrier"):
                color = label_to_colors.get(c, n.carriers.color.get(c, ""))
                if not isinstance(color, str) or color == "":
                    color = "lightgrey"
                carrier_colors.setdefault(c, color)
        panels.append(region_panels)

    def get_label(c):
        return c.replace(" renewable cluster", "")

    # split the technologies in supply and consumption; the ones appearing
    # on both sides go where they are larger (summed over all the maps)
    all_sizes = pd.concat([bus_size for region_panels in panels for _, _, bus_size in region_panels])
    supply_total = all_sizes[all_sizes > 0].groupby(level="carrier").sum()
    consumption_total = all_sizes[all_sizes < 0].abs().groupby(level="carrier").sum()
    supp_carriers = sorted(
        c for c in carrier_colors
        if supply_total.get(c, 0) >= consumption_total.get(c, 0)
    )
    cons_carriers = sorted(c for c in carrier_colors if c not in supp_carriers)

    # --------------------------------------------------
    # Map extents with a common height / width ratio
    # --------------------------------------------------
    config = config_plotting["plotting"]
    crs = load_projection(copy.deepcopy(config))

    extents = [get_projected_extent(crs, case["boundaries"]) for case in region_cases]
    aspects = [(e[3] - e[2]) / (e[1] - e[0]) for e in extents]
    aspect = float(np.exp(np.mean(np.log(aspects))))
    extents = [pad_extent_to_aspect(e, aspect) for e in extents]

    # --------------------------------------------------
    # Figure layout (all lengths in inches)
    # --------------------------------------------------
    ncols = len(region_cases)
    nrows = max(len(region_panels) for region_panels in panels)
    # font sizes as in plot_co2_stored_balance_maps, headers slightly larger
    title_size = 17
    panel_title_size = 11
    legend_fontsize = 9
    legend_title_fontsize = 10

    left, right, bottom = 0.1, 0.1, 0.1
    header_h = 0.45  # region titles above the columns
    panel_title_h = 0.3  # combination titles above each map
    wspace, hspace = 0.25, 0.05
    title_pad = 0.15  # minimum gap between the titles of neighbouring maps [in]
    legend_pad = 0.1  # gap between the maps and the legends [in]
    legend_gap = 0.3  # gap between the legends [in]

    tech_rows = int(
        max(
            np.ceil(len(supp_carriers) / supply_ncol),
            np.ceil(len(cons_carriers) / consumption_ncol),
        )
    )
    tech_legend_h = (legend_title_fontsize * 1.6 + tech_rows * legend_fontsize * 1.5) / 72

    def legend_height(map_w):
        # radius of the legend semicircles in points (checked against the
        # circles actually drawn on the maps further down)
        max_radius = max(
            (np.sqrt(s) * circle_scale * map_w * 72 for s in legend_sizes), default=0
        )
        size_legend_h = (
            legend_title_fontsize * 1.6
            + len(legend_sizes) * (max_radius + legend_fontsize)
        ) / 72
        return max(tech_legend_h, size_legend_h) + 0.1

    def figure_height(map_w):
        maps_h = nrows * (panel_title_h + map_w * aspect) + (nrows - 1) * hspace
        return header_h + maps_h + legend_pad + legend_height(map_w) + bottom

    # maps as wide as the figure allows; narrower if the figure would be
    # taller than max_fig_height (the legend shrinks with the maps, so the
    # second pass keeps the figure within the limit)
    map_w = (fig_width - left - right - wspace * (ncols - 1)) / ncols
    for _ in range(2):
        excess = figure_height(map_w) - max_fig_height
        if excess > 0:
            map_w -= excess / (nrows * aspect)

    # columns far enough apart for the titles of neighbouring maps not to
    # overlap (widest combination title measured with the figure fonts);
    # the maps get narrower if the columns would then not fit the width
    probe = plt.figure()
    probe_renderer = probe.canvas.get_renderer()
    title_w = max(
        probe.text(0, 0, italic_combination_label(key), fontsize=panel_title_size)
        .get_window_extent(probe_renderer).width / probe.dpi
        for region_panels in panels for key, _, _ in region_panels
    )
    plt.close(probe)
    wspace = max(wspace, title_w + title_pad - map_w)
    map_w = min(map_w, (fig_width - left - right - wspace * (ncols - 1)) / ncols)

    map_h = map_w * aspect
    legend_h = legend_height(map_w)
    fig_height = figure_height(map_w)

    # figure only as wide as the maps, plus the part of the titles of the
    # outer maps sticking out of them (the legends below use the same width)
    maps_w = ncols * map_w + (ncols - 1) * wspace
    left = right = max(left, (title_w - map_w) / 2 + title_pad / 2)
    maps_left = left
    fig_width = maps_w + left + right

    row_h = panel_title_h + map_h
    # every axes is placed by hand, so no automatic layout (it is switched on
    # by the 'figure.autolayout' setting in some environments)
    fig = plt.figure(figsize=(fig_width, fig_height), layout="none")

    axes = []  # per region: list of axes
    for col, region_panels in enumerate(panels):
        x0 = maps_left + col * (map_w + wspace)
        fig.text(
            (x0 + map_w / 2) / fig_width,
            1 - header_h / 2 / fig_height,
            region_cases[col].get("title", ""),
            fontsize=title_size,
            ha="center",
            va="center",
        )
        region_axes = []
        for row in range(len(region_panels)):
            map_top = fig_height - header_h - panel_title_h - row * (row_h + hspace)
            region_axes.append(
                fig.add_axes(
                    [
                        x0 / fig_width,
                        (map_top - map_h) / fig_height,
                        map_w / fig_width,
                        map_h / fig_height,
                    ],
                    projection=crs,
                )
            )
        axes.append(region_axes)

    # --------------------------------------------------
    # Maps
    # --------------------------------------------------
    for case, region_panels, region_axes, extent in zip(region_cases, panels, axes, extents):
        regions_crs = case["regions"].to_crs(crs.proj4_init)

        for ax, (key, n, bus_size) in zip(region_axes, region_panels):
            ax.set_extent(extent, crs=crs)
            ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
            ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
            # region borders between the nodes
            regions_crs.plot(
                ax=ax,
                facecolor="none",
                edgecolor="lightgrey",
                linewidth=0.3,
                zorder=0.5,
            )
            ax.add_feature(cfeature.COASTLINE, edgecolor="darkgrey", linewidth=0.5, zorder=1)
            ax.add_feature(cfeature.BORDERS, edgecolor="darkgrey", linewidth=0.3, zorder=1)
            ax.spines['geo'].set_visible(False)
            ax.set_extent(extent, crs=crs)

            # radius of a 1 `unit` semicircle in the units of the projection
            unit_radius = circle_scale * (extent[1] - extent[0])

            for bus, group in bus_size.groupby(level="bus"):
                group = group.droplevel("bus")
                x = float(n.buses.at[bus, "x"])
                y = float(n.buses.at[bus, "y"])
                x_proj, y_proj = ax.projection.transform_point(x, y, ccrs.PlateCarree())

                # upper half: supply (theta 180 -> 0), lower half: consumption (theta 180 -> 360)
                for values, direction in [
                    (group[group > 0], -1),
                    (group[group < 0].abs(), 1),
                ]:
                    total = values.sum()
                    if total <= 0:
                        continue

                    radius = np.sqrt(total) * unit_radius
                    values = values.sort_values(ascending=False)

                    theta1 = 180
                    for tech, value in values.items():
                        theta2 = theta1 + direction * value / total * 180

                        angles = np.linspace(np.radians(theta1), np.radians(theta2), 100)
                        xs = x_proj + radius * np.cos(angles)
                        ys = y_proj + radius * np.sin(angles)
                        verts = np.column_stack([xs, ys])
                        verts = np.vstack([[x_proj, y_proj], verts, [x_proj, y_proj]])

                        poly = Polygon(
                            verts,
                            closed=True,
                            facecolor=carrier_colors[tech],
                            edgecolor="white",
                            linewidth=0.3,
                            zorder=3,
                        )
                        ax.add_patch(poly)
                        theta1 = theta2

            ax.set_title(italic_combination_label(key), fontsize=panel_title_size)

    # --------------------------------------------------
    # Check of the circle sizes: measure, on the rendered maps, the radius
    # in points of a 1 `unit` semicircle and use it for the legend semicircles
    # --------------------------------------------------
    fig.canvas.draw()
    points_per_sqrt_unit = []
    for region_axes, extent in zip(axes, extents):
        for ax in region_axes:
            x_pixels = ax.transData.transform(
                [(0, 0), (circle_scale * (extent[1] - extent[0]), 0)]
            )[:, 0]
            points_per_sqrt_unit.append((x_pixels[1] - x_pixels[0]) * 72 / fig.dpi)
    if not np.allclose(points_per_sqrt_unit, circle_scale * map_w * 72, rtol=0.01):
        print(
            "WARNING: the maps are not drawn at the expected size, radius of a "
            f"1 {unit} semicircle in points: expected {circle_scale * map_w * 72:.2f}, "
            f"drawn {np.round(points_per_sqrt_unit, 2)}"
        )
    legend_radii = [np.sqrt(s) * points_per_sqrt_unit[0] for s in legend_sizes]
    max_radius = max(legend_radii, default=0)

    # --------------------------------------------------
    # Legends (shared, below the maps):
    # circle sizes to the left, supply and consumption technologies to the right
    # --------------------------------------------------
    lax = fig.add_axes(
        [
            left / fig_width,
            bottom / fig_height,
            (fig_width - left - right) / fig_width,
            legend_h / fig_height,
        ]
    )
    lax.axis("off")

    legend_kwargs = {
        "loc": "upper left",
        "frameon": False,
        "alignment": "left",
        "fontsize": legend_fontsize,
        "title_fontproperties": {
            "weight": "bold",
            "size": legend_title_fontsize,
        },
        "borderaxespad": 0,
    }

    legends = []
    if len(legend_sizes) > 0:
        size_legend = lax.legend(
            [Wedge((0, 0), r, 0, 180, facecolor="#666", edgecolor="None") for r in legend_radii],
            [f"{s:g} {unit}" for s in legend_sizes],
            handler_map={Wedge: HandlerSemicircle()},
            bbox_to_anchor=(0, 1),
            title="Circle size",
            handlelength=2 * max_radius / legend_fontsize,
            handleheight=max_radius / legend_fontsize,
            labelspacing=0.6,
            **legend_kwargs,
        )
        lax.add_artist(size_legend)
        legends.append(size_legend)

    tech_legend_kwargs = {
        "handlelength": 1.0,
        "handleheight": 1.0,
        "handletextpad": 0.4,
        "labelspacing": 0.25,
        "columnspacing": 1.2,
        **legend_kwargs,
    }

    supply_legend = lax.legend(
        [Patch(facecolor=carrier_colors[t], edgecolor="None") for t in supp_carriers],
        [get_label(t) for t in supp_carriers],
        bbox_to_anchor=(0.2, 1),
        title="Supply",
        ncol=supply_ncol,
        **tech_legend_kwargs,
    )
    lax.add_artist(supply_legend)
    legends.append(supply_legend)

    consumption_legend = lax.legend(
        [Patch(facecolor=carrier_colors[t], edgecolor="None") for t in cons_carriers],
        [get_label(t) for t in cons_carriers],
        bbox_to_anchor=(0.5, 1),
        title="Consumption",
        ncol=consumption_ncol,
        **tech_legend_kwargs,
    )
    legends.append(consumption_legend)

    # keep the legends close together, as a group starting at the left edge
    # of the maps and with the titles on the same line (sizes and positions
    # measured once drawn); the gap shrinks if the group would not fit
    fig.canvas.draw()
    lax_box = lax.get_window_extent()
    widths = [leg.get_window_extent().width / lax_box.width for leg in legends]
    title_tops = [leg.get_title().get_window_extent().y1 for leg in legends]
    n_gaps = max(len(legends) - 1, 1)
    gap = min(legend_gap / (fig_width - left - right), (1 - sum(widths)) / n_gaps)
    x = 0
    for leg, width, title_top in zip(legends, widths, title_tops):
        y = 1 - (title_top - min(title_tops)) / lax_box.height
        leg.set_bbox_to_anchor((x, y), transform=lax.transAxes)
        x += width + gap

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return [
        {key: bus_size for key, _, bus_size in region_panels} for region_panels in panels
    ]





#%%
# Build the combined metrics DataFrame once: each network (both regions) is
# loaded from disk exactly once, and every plot below (single-region and
# combined) reuses this in-memory DataFrame instead of reloading networks.

iberian_networks_folder = r"results/Iberic40_2035_industrial_clusters_all3hrerun/all/networks"
nordic_networks_folder = r"results/Noridcs100_2035_industrial_clusters_all3hrerun/all/networks"

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
    # ("CO₂ Captured Cluster [Mtons/year]", "Purples"),
    # ("CO₂ Price [€/ton]", "Reds"),
    # ("Renewable Capacity Cluster [GW]", "Greens"),
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
    # ("CO₂ Price [€/ton]", "Reds"),
    # ("Renewable Capacity Cluster [GW]", "Greens"),
     ("Share of Curtailment in the Cluster[%]", YELLOWS),
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
    fig.savefig(PICTURES_DIR / f"heatmap_curtailment_{axis}.pdf", bbox_inches="tight")
    plt.show()

#%%
# Per-node capacities and full load hours of the renewable cluster components


CLUSTER_GENERATORS = ["solar", "solar-hsat", "onwind"]
CLUSTER_LINKS = ["H2 Electrolysis", "urban decentral DAC", "methanolisation", "Fischer-Tropsch"]
# Grid connection links: PyPSA link name suffix -> tech name used in the plots.
# Their p_nom_opt is already in MW of electricity.
CLUSTER_GRIDS = {
    'electricity renewable cluster': 'grid connection (sell) renewable cluster',
    'electricity renewable cluster back': 'grid connection (buy) renewable cluster',
    'electricity renewable cluster both': 'grid connection (both) renewable cluster',
}
CLUSTER_TECHS = CLUSTER_GENERATORS + CLUSTER_LINKS + list(CLUSTER_GRIDS.values())



# (CR, BUYcap, SELLcap, BOTHcap)
combinations = [
    (0.3, 0, 0, 0), #reference case
    (0.3, 0.25, 0,0),
    (0.4, 0, 0, 0),
    (0.4, 0.25, 0, 0),
    (0.5, 0, 0, 0),
    (0.5, 0.25, 0, 0),
]

renewable_cluster_capacities_iberian, renewable_cluster_full_load_hours_iberian = build_cluster_dicts(
    df_iberian, combinations, 50, top_n_tech = CLUSTER_LINKS
)
renewable_cluster_capacities_nordic, renewable_cluster_full_load_hours_nordic = build_cluster_dicts(
    df_nordic, combinations, 50, top_n_tech = CLUSTER_LINKS
)




# Region name -> (capacities dict, full load hours dict), used by the plots below
CLUSTER_RESULTS = {
    REGION_IBERIAN: (renewable_cluster_capacities_iberian, renewable_cluster_full_load_hours_iberian),
    REGION_NORDIC: (renewable_cluster_capacities_nordic, renewable_cluster_full_load_hours_nordic),
}

for region, (caps_dict, flh_dict) in CLUSTER_RESULTS.items():
    fig, axes = plot_cluster_capacities_and_flh(caps_dict, flh_dict, region=region, log_capacity=False,
                                                offgrid_key=combinations[0], min_gen_capacity=50.0, min_link_capacity=10.0)
    plt.show()


for region, (caps_dict, flh_dict) in CLUSTER_RESULTS.items():
    fig, axes = plot_cluster_boxplots(caps_dict, flh_dict, region=region, log_capacity=False)
    plt.show()

#%%
n = load_network(iberian_networks_folder, 0.4, 0.5, 0, 0) 
#%%
# AC balance maps of the renewable cluster: reference loading hours in the
# first panel, difference of loading hours vs. the reference in the others

# (CR, BUYcap, SELLcap, BOTHcap)
combinations = [
    (0.0, 0, 0.1, 0),
    (0.0, 0, 0, 0.1),
    (0.1, 0, 0.1, 0),
    (0.1, 0, 0, 0.1),
]

fig, axes, bus_sizes_iberian = plot_cluster_electricity_balance_map_combinations(
    df_iberian, combinations, regions_iberian, config_plotting, label_to_colors,
    boundaries=[-15, 12, 35, 48],
)
plt.show()

fig, axes, bus_sizes_nordic = plot_cluster_electricity_balance_map_combinations(
    df_nordic, combinations, regions_nordic, config_plotting, label_to_colors,
    boundaries=[-10,28, 46,73],
)
plt.show()

#%%
# Same maps with the lines coloured by the mean absolute shadow price of
# their capacity constraint (reference) and its difference vs. the reference

# (CR, BUYcap, SELLcap, BOTHcap)
combinations = [
    (0.0, 0, 0.0, 0),
    (0.4, 0, 0, 0),    
    (0.1, 0, 0, 0.1),    
    (0.4, 0, 0, 0.1),
    (0.0, 0, 0.0, 0.2),

]

fig, axes, bus_sizes_iberian = plot_cluster_electricity_balance_map_combinations(
    df_iberian, combinations, regions_iberian, config_plotting, label_to_colors,
    boundaries=[-15, 12, 35, 48],
    branch_func=branch_mean_shadow_price,
    branch_label="Mean line shadow price", branch_unit="€/MVA",
)
plt.show()

fig, axes, bus_sizes_nordic = plot_cluster_electricity_balance_map_combinations(
    df_nordic, combinations, regions_nordic, config_plotting, label_to_colors,
    boundaries=[-10,28, 46,73],
    branch_func=branch_mean_shadow_price,
    branch_label="Mean line shadow price", branch_unit="€/MVA",
)
plt.show()


#%%
# AC balance maps of the renewable cluster: reference demand-weighted
# electricity price on the nodes in the first panel, difference of that
# price vs. the reference in the others

# (CR, BUYcap, SELLcap, BOTHcap)
combinations = [
    (0.0, 0, 0.0, 0),
    (0.4, 0, 0, 0),    
    (0.1, 0, 0, 0.1),    
    (0.4, 0, 0, 0.1),
    (0.0, 0, 0.0, 0.2),

]

fig, axes, bus_sizes_iberian, prices_iberian = plot_cluster_electricity_balance_price_map_combinations(
    df_iberian, combinations, regions_iberian, config_plotting, label_to_colors,
    boundaries=[-15, 12, 35, 48],
)
plt.show()

fig, axes, bus_sizes_nordic, prices_nordic = plot_cluster_electricity_balance_price_map_combinations(
    df_nordic, combinations, regions_nordic, config_plotting, label_to_colors,
    boundaries=[-10,28, 46,73],
)
plt.show()

#%%
# Same maps without comparison: every panel coloured by its own price on the
# same scale, one colorbar on the right of the figure

# (CR, BUYcap, SELLcap, BOTHcap)
combinations = [
    (0.0, 0, 0.0, 0),
    (0.4, 0, 0, 0),    
    (0.1, 0, 0, 0.1),    
    (0.4, 0, 0, 0.1),
    (0.0, 0, 0.0, 0.2),

]

fig, axes, bus_sizes_iberian, prices_iberian = plot_cluster_electricity_balance_price_map_combinations(
    df_iberian, combinations, regions_iberian, config_plotting, label_to_colors,
    boundaries=[-15, 12, 35, 48],
    color_mode="absolute",
)
plt.show()

fig, axes, bus_sizes_nordic, prices_nordic = plot_cluster_electricity_balance_price_map_combinations(
    df_nordic, combinations, regions_nordic, config_plotting, label_to_colors,
    boundaries=[-10,28, 46,73],
    color_mode="absolute",
)
plt.show()

#%%
# Same maps without any price colouring

# (CR, BUYcap, SELLcap, BOTHcap)
combinations = [
    (0.4, 0, 0, 0),    
    (0.1, 0, 0, 0.1),    
    (0.4, 0, 0, 0.1),
    (0.0, 0, 0.0, 0.2),

]
fig, axes, bus_sizes_iberian, prices_iberian = plot_cluster_electricity_balance_price_map_combinations(
    df_iberian, combinations, regions_iberian, config_plotting, label_to_colors,
    boundaries=[-15, 12, 35, 48],
    color_mode="none",
    show_branches=False,
)
plt.show()

# (CR, BUYcap, SELLcap, BOTHcap)
combinations = [
    (0.3, 0, 0, 0),    
    (0, 0, 0, 0.1),    
    (0.4, 0, 0, 0.1),
    (0.0, 0, 0.0, 0.2),

]

fig, axes, bus_sizes_nordic, prices_nordic = plot_cluster_electricity_balance_price_map_combinations(
    df_nordic, combinations, regions_nordic, config_plotting, label_to_colors,
    boundaries=[-10,20, 46,70],
    color_mode="none",
    show_branches=False,
)
plt.show()

#%%
# Same maps with the maximum electricity price of the year
fig, axes, bus_sizes_iberian, max_prices_iberian = plot_cluster_electricity_balance_max_price_map_combinations(
    df_iberian, combinations, regions_iberian, config_plotting, label_to_colors,
    boundaries=[-15, 12, 35, 48],
)
plt.show()

fig, axes, bus_sizes_nordic, max_prices_nordic = plot_cluster_electricity_balance_max_price_map_combinations(
    df_nordic, combinations, regions_nordic, config_plotting, label_to_colors,
    boundaries=[-10,28, 46,73],
)
plt.show()

#%%
# Same maps with the minimum electricity price of the year
fig, axes, bus_sizes_iberian, min_prices_iberian = plot_cluster_electricity_balance_min_price_map_combinations(
    df_iberian, combinations, regions_iberian, config_plotting, label_to_colors,
    boundaries=[-15, 12, 35, 48],
)
plt.show()

fig, axes, bus_sizes_nordic, min_prices_nordic = plot_cluster_electricity_balance_min_price_map_combinations(
    df_nordic, combinations, regions_nordic, config_plotting, label_to_colors,
    boundaries=[-10,28, 46,73],
)
plt.show()
# %%
# Electricity balance of the renewable clusters of both regions in one figure,
# with the same circle scale: Iberian Peninsula on the left, Nordic Countries
# on the right, one row per combination
bus_sizes_two_regions = plot_cluster_electricity_balance_maps_two_regions(
    [
        {
            "df": df_iberian,
            # (CR, BUYcap, SELLcap, BOTHcap)
            "combinations": [
                # (0.4, 0, 0, 0),
                # (0.1, 0, 0, 0.1),
                # (0.0, 0, 0, 0.2),
                # (0.4, 0.25, 0, 0),
                # (0.1, 0, 0.1, 0),
                (0.4, 0, 0, 0.1),
                (0.4, 0, 0.1, 0),
                (0.0, 0, 0.2, 0),
            ],
            "regions": regions_iberian,
            # Spain and Portugal
            "boundaries": [-10.5, 5, 35.5, 44.5],
            "title": "Iberian Peninsula",
        },
        {
            "df": df_nordic,
            # (CR, BUYcap, SELLcap, BOTHcap)
            "combinations": [
                (0.3, 0, 0, 0),
                (0, 0, 0, 0.1),
                (0.0, 0, 0, 0.2),
                # (0.3, 0.25, 0, 0),
                # (0, 0, 0.1, 0),
                # (0.4, 0, 0.1, 0),
                # (0.4, 0, 0.1, 0),
                # (0.0, 0, 0.2, 0),
            ],
            "regions": regions_nordic,
            # Germany, Denmark, UK, Netherlands, Sweden and Norway
            "boundaries": [-10, 20, 47, 70],
            "title": "North Sea Region",
        },
    ],
    config_plotting,
    label_to_colors,
    save_path=PICTURES_DIR / "ClusterElectricityBalanceMaps_TwoRegions.pdf",
)

# %%
