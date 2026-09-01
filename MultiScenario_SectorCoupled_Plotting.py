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
    'solid biomass biomass-to-methanol': "#8442f5",
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
    'Sabatier': '#de6012',

    # Renewable cluster variants
    'methanolisation renewable cluster': '#36ad0e',
    'H2 Electrolysis renewable cluster': "#42f5d1",
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


def add_co2_marginal_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load each network from its path and add a column with the CO2 marginal
    price [euro/ton], taken from the shadow price (mu) of the CO2Limit global
    constraint.
    """
    df = df.copy()
    co2_price = []
    for path in df["path"]:
        n = pypsa.Network(str(path))
        mask = n.global_constraints.index.str.contains("CO2Limit")
        co2_price.append(abs(n.global_constraints.loc[mask, "mu"].iloc[0]))
    df["co2 price [euro/ton]"] = co2_price
    return df


def add_renewable_capacity_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load each network from its path and add a column with the total optimised
    renewable generator capacity installed in the industrial clusters [MW].
    """
    df = df.copy()
    renewable_capacity = []
    for path in df["path"]:
        n = pypsa.Network(str(path))
        mask = n.generators.index.str.contains("renewable cluster")
        renewable_capacity.append(n.generators.loc[mask, "p_nom_opt"].sum())
    df["renewable capacity cluster [MW]"] = renewable_capacity
    return df


def plot_heatmap_buycap_vs_cr(
    df: pd.DataFrame,
    sellcap,
    bothcap,
    value_col,
    cmap="Purples",
    cr_range=None,
    buycap_range=None,
    title="CO2 captured in cluster [Mtons/year]",
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
    ax.set_title(title)
    return fig, ax


def plot_heatmap_sellcap_vs_cr(
    df: pd.DataFrame,
    buycap,
    bothcap,
    value_col,
    cmap="Purples",
    cr_range=None,
    sellcap_range=None,
    title="CO2 captured in cluster [Mtons/year]",
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
    ax.set_title(title)
    return fig, ax

def plot_heatmap_bothcap_vs_cr(
    df: pd.DataFrame,
    buycap,
    sellcap,
    value_col,
    cmap="Purples",
    cr_range=None,
    bothcap_range=None,
    title="CO2 captured in cluster [Mtons/year]",
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
    ax.set_title(title)
    return fig, ax

def compute_ft_and_methanol_prices(n):
    """
    Compute FT oil and methanol prices per node based on marginal costs
    of inputs (H2, CO2, electricity, heat) and allocated capital costs.

    Parameters
    ----------
    n : pypsa.Network
        The solved network (generic name, e.g. n_base).

    Returns
    -------
    FT_oil_prices : pd.DataFrame
    methanol_prices : pd.DataFrame
    FT_oil_prices_average : pd.Series (indexed by location)
    methanol_prices_average : pd.Series (indexed by location)
    """

    weights = n.snapshot_weightings.generators

    locations = n.buses.loc[ (n.buses['carrier'] == 'AC'),['location']].location.unique()

    methanol_prices = pd.DataFrame(index=n.snapshots, columns=locations)
    FT_oil_prices = pd.DataFrame(index=n.snapshots, columns=locations)
    capital_cost_allocated_FT = pd.DataFrame(index=n.snapshots, columns=locations)
    capital_cost_allocated_methanolisation = pd.DataFrame(index=n.snapshots, columns=locations)

    FT_oil_prices_average = pd.Series(index=locations, dtype=float)
    methanol_prices_average = pd.Series(index=locations, dtype=float)

    co2_intensity_FT = (
        n.links.loc[n.links.index.str.contains("Fischer-Tropsch"), 'efficiency2'].iloc[0]
        / n.links.loc[n.links.index.str.contains("Fischer-Tropsch"), 'efficiency'].iloc[0]
    )
    print(f"co2_intensity_FT: {co2_intensity_FT}")

    co2_intensity_methanol = n.links.loc[
        n.links.index.str.contains("EU shipping methanol"), 'efficiency2'
    ].iloc[0]
    print(f"co2_intensity_methanol: {co2_intensity_methanol}")

    co2_price = n.global_constraints.loc[
        n.global_constraints.carrier_attribute.str.contains("co2_emissions"), 'mu'
    ].iloc[0]

    for node in locations:

        # ---------------- Fischer-Tropsch ----------------
        if n.links.index.str.contains(f"^{node} Fischer-Tropsch$").any():

            if n.links.loc[n.links.index == f"{node} Fischer-Tropsch", 'p_nom_opt'].iloc[0] > 1:
                O_and_M_cost = n.links.loc[n.links.index == f"{node} Fischer-Tropsch", 'marginal_cost'].iloc[0]
                capital_cost = (
                    n.links.loc[n.links.index == f"{node} Fischer-Tropsch", 'capital_cost'].iloc[0]
                    * n.links.loc[n.links.index == f"{node} Fischer-Tropsch", 'p_nom_opt'].iloc[0]
                )

                price_of_H2 = n.buses_t.marginal_price.loc[:, node + ' H2']
                consumption_of_H2 = n.links_t.p0.loc[:, node + ' Fischer-Tropsch']
                consumption_of_H2_total = n.links_t.p0.loc[:, node + ' Fischer-Tropsch'].multiply(weights, axis=0).sum()

                capital_cost_allocated_FT[node] = capital_cost * consumption_of_H2 / consumption_of_H2_total

                price_of_CO2 = n.buses_t.marginal_price.loc[:, node + ' co2 stored']
                consumption_of_CO2 = n.links_t.p2.loc[:, node + ' Fischer-Tropsch']

                price_of_heat = 0
                production_of_heat = 0
                if node + ' urban central heat' in n.buses_t.marginal_price.columns:
                    price_of_heat = n.buses_t.marginal_price.loc[:, node + ' urban central heat']
                    production_of_heat = n.links_t.p3.loc[:, node + ' Fischer-Tropsch'].abs()

                production_of_FT_oil = n.links_t.p1.loc[:, node + ' Fischer-Tropsch'].abs()

                price_of_FT_oil = (
                    price_of_H2 * consumption_of_H2
                    + O_and_M_cost * consumption_of_H2
                    + capital_cost_allocated_FT[node]
                    + price_of_CO2 * consumption_of_CO2
                    - price_of_heat * production_of_heat
                ) / production_of_FT_oil - (-co2_intensity_FT * co2_price)

                FT_oil_prices[node] = price_of_FT_oil

                FT_oil_prices_average[node] = (
                    price_of_H2 * consumption_of_H2
                    + O_and_M_cost * consumption_of_H2
                    + capital_cost_allocated_FT[node]
                    + price_of_CO2 * consumption_of_CO2
                    - price_of_heat * production_of_heat
                ).sum() / production_of_FT_oil.sum() - (-co2_intensity_FT * co2_price)

        # ---------------- Methanolisation ----------------
        if n.links.index.str.contains(f"^{node} methanolisation$").any():

            if n.links.loc[n.links.index == f"{node} methanolisation", 'p_nom_opt'].iloc[0] > 1:
                O_and_M_cost = n.links.loc[n.links.index == f"{node} methanolisation", 'marginal_cost'].iloc[0]
                capital_cost = (
                    n.links.loc[n.links.index == f"{node} methanolisation", 'capital_cost'].iloc[0]
                    * n.links.loc[n.links.index == f"{node} methanolisation", 'p_nom_opt'].iloc[0]
                )

                price_of_H2 = n.buses_t.marginal_price.loc[:, node + ' H2']
                consumption_of_H2 = n.links_t.p0.loc[:, node + ' methanolisation']
                consumption_of_H2_total = n.links_t.p0.loc[:, node + ' methanolisation'].multiply(weights, axis=0).sum()

                capital_cost_allocated_methanolisation[node] = (
                    capital_cost * consumption_of_H2 / consumption_of_H2_total
                )

                price_of_CO2 = n.buses_t.marginal_price.loc[:, node + ' co2 stored']
                consumption_of_CO2 = n.links_t.p3.loc[:, node + ' methanolisation']
                price_of_electricity = n.buses_t.marginal_price.loc[:, node]
                consumption_of_electricity = n.links_t.p2.loc[:, node + ' methanolisation']

                price_of_heat = 0
                production_of_heat = 0
                if node + ' urban central heat' in n.buses_t.marginal_price.columns:
                    price_of_heat = n.buses_t.marginal_price.loc[:, node + ' urban central heat']
                    production_of_heat = n.links_t.p4.loc[:, node + ' methanolisation'].abs()

                production_of_methanol = n.links_t.p1.loc[:, node + ' methanolisation'].abs()

                price_of_methanol = (
                    price_of_H2 * consumption_of_H2
                    + O_and_M_cost * consumption_of_H2
                    + capital_cost_allocated_methanolisation[node]
                    + price_of_electricity * consumption_of_electricity
                    + price_of_CO2 * consumption_of_CO2
                    - price_of_heat * production_of_heat
                ) / production_of_methanol - co2_intensity_methanol * co2_price

                methanol_prices[node] = price_of_methanol

                methanol_prices_average[node] = (
                    price_of_H2 * consumption_of_H2
                    + O_and_M_cost * consumption_of_H2
                    + capital_cost_allocated_methanolisation[node]
                    + price_of_electricity * consumption_of_electricity
                    + price_of_CO2 * consumption_of_CO2
                    - price_of_heat * production_of_heat
                ).sum() / production_of_methanol.sum() - co2_intensity_methanol * co2_price


    return FT_oil_prices, methanol_prices, FT_oil_prices_average, methanol_prices_average 
    

def plot_balance_map_methanol_price(
    n,
    regions,
    config_plotting,
    label_to_colors,
    boundaries,
):


    SEMICIRCLE_CORRECTION_FACTOR = (
        2 if parse(pypsa.__version__) <= Version("0.33.2")
        else 1
    )
    carrier = "co2 stored"
    config = config_plotting["plotting"]
    settings = config["balance_map"]["co2 stored"]
    if carrier not in n.buses.carrier.unique():
        raise ValueError(
            f"Carrier '{carrier}' is not in the network."
        )
    # --------------------------------------------------
    # Carrier colours
    # --------------------------------------------------
    mask = n.carriers.color.isna() | n.carriers.color.eq("")
    n.carriers["color"] = n.carriers.color.mask(
        mask,
        "lightgrey",
    )
    n.carriers.loc["", "color"] = "None"
    for label, color in label_to_colors.items():
        if label in n.carriers.index:
            n.carriers.loc[label, "color"] = color
    # --------------------------------------------------
    # EU bus location
    # --------------------------------------------------
    eu_location = config["eu_node_location"]
    n.buses.loc["EU", ["x", "y"]] = (
        eu_location["x"],
        eu_location["y"],
    )
    # --------------------------------------------------
    # Project buses to locations
    # --------------------------------------------------
    n.buses["location"] = (
        n.buses["location"]
        .replace("", "EU")
        .fillna("EU")
    )
    n.buses["x"] = n.buses.location.map(n.buses.x)
    n.buses["y"] = n.buses.location.map(n.buses.y)
    # --------------------------------------------------
    # Energy balance
    # --------------------------------------------------
    pypsa.set_option("params.statistics.round", 8)
    pypsa.set_option("params.statistics.drop_zero", True)
    pypsa.set_option("params.statistics.nice_names", False)
    unit_conversion = settings["unit_conversion"]
    eb = n.statistics.energy_balance(
        bus_carrier=carrier,
        groupby=["bus", "carrier"],
    )
    # collapse the two CO2 storage buses into one
    new_bus = (
        eb.index.get_level_values("bus")
        .str.replace(" co2 stored DAC", "", regex=False)
        .str.replace(" co2 stored industrial", "", regex=False)
        .str.replace(" co2 stored", "", regex=False)
    )
    eb.index = pd.MultiIndex.from_arrays(
        [
            eb.index.get_level_values("component"),
            new_bus,
            eb.index.get_level_values("carrier"),
        ],
        names=eb.index.names,
    )
    # aggregate duplicate buses
    eb = eb.groupby(level=["component", "bus", "carrier"]).sum()
    transmission_carriers = (
        get_transmission_carriers(
            n,
            bus_carrier=carrier,
        )
        .rename({"name": "carrier"})
    )
    components = transmission_carriers.unique("component")
    carriers = transmission_carriers.unique("carrier")
    carriers_in_eb = carriers[
        carriers.isin(
            eb.index.get_level_values("carrier")
        )
    ]
    eb.loc[components] = (
        eb.loc[components]
        .drop(
            index=carriers_in_eb,
            level="carrier",
        )
    )
    eb = eb.dropna()
    bus_size = (
        eb.groupby(level=["bus", "carrier"])
        .sum()
        .div(unit_conversion)
        .sort_values(ascending=False)
    )
    # --------------------------------------------------
    # Bus colours
    # --------------------------------------------------
    carrier_colors = (
        n.carriers.color.copy()
        .replace("", "grey")
    )
    for label, color in label_to_colors.items():
        if label in carrier_colors.index:
            carrier_colors[label] = color
    colors = (
        bus_size.index.get_level_values("carrier")
        .unique()
        .to_series()
        .map(carrier_colors)
        .fillna("lightgrey")
    )
    bus_size_factor = settings["bus_factor"]
    # ==================================================
    # AVERAGE METHANOL PRICE (per node, time-averaged)
    # ==================================================
    _, _, _, methanol_prices_average = compute_ft_and_methanol_prices(n)
    avg_methanol_price = pd.to_numeric(
        methanol_prices_average, errors="coerce"
    ).round(1)
    regions = regions.copy()
    # NOTE: no fillna here - nodes with no methanol price stay NaN
    # so geopandas skips them instead of drawing them as 0.
    regions["methanol_price"] = avg_methanol_price.reindex(regions.index)

    vmin = regions.methanol_price.min()  # NaNs ignored automatically
    vmax = regions.methanol_price.max()
    if vmin == vmax:
        vmax += 1
    # --------------------------------------------------
    # Teal colormap
    # --------------------------------------------------
    teal_cmap = mcolors.LinearSegmentedColormap.from_list(
        "teal_scale",
        ["#e5f5f4", "#00706b", "#00312e"],
    )
    # --------------------------------------------------
    # Figure
    # --------------------------------------------------
    crs = load_projection(copy.deepcopy(config))
    fig, ax = plt.subplots(
        figsize=(5, 6.5),
        subplot_kw={"projection": crs},
        layout="constrained",
    )
    # --------------------------------------------------
    # Plot regional colormap
    # --------------------------------------------------
    regions.to_crs(crs.proj4_init).plot(
        ax=ax,
        column="methanol_price",
        cmap=teal_cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolor="None",
        linewidth=0,
        zorder=0,
        missing_kwds={
            "color": "none",
        },
    )
    # --------------------------------------------------
    # Plot PyPSA circles
    # --------------------------------------------------
    n.plot(
        ax=ax,
        bus_size=bus_size * bus_size_factor,
        bus_color=colors,
        bus_split_circle=True,
        line_width=0,
        link_width=0,
        margin=0.2,
        geomap=True,
        geomap_color={
            "border": "darkgrey",
            "coastline": "darkgrey",
        },
        boundaries=boundaries,
    )
    ax.set_title("CO₂ stored")
    # --------------------------------------------------
    # Colorbar
    # --------------------------------------------------
    norm = plt.Normalize(
        vmin=vmin,
        vmax=vmax,
    )
    sm = plt.cm.ScalarMappable(
        cmap=teal_cmap,
        norm=norm,
    )
    cbar = fig.colorbar(
        sm,
        ax=ax,
        label="Average methanol price [€/MWh]",
        shrink=0.95,
        pad=0.03,
        aspect=50,
        orientation="horizontal",
    )
    from matplotlib.ticker import FormatStrFormatter

    cbar.outline.set_edgecolor("None")
    cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # --------------------------------------------------
    # Legends
    # --------------------------------------------------
    legend_kwargs = {
        "loc": "upper left",
        "frameon": False,
        "alignment": "left",
        "title_fontproperties": {
            "weight": "bold"
        },
    }
    pad = 0.18
    pos_carriers = (
        bus_size[bus_size > 0]
        .index.unique("carrier")
    )
    neg_carriers = (
        bus_size[bus_size < 0]
        .index.unique("carrier")
    )
    common_carriers = (
        pos_carriers.intersection(
            neg_carriers
        )
    )

    def get_total_abs(c, sign):
        vals = bus_size.loc[:, c]
        return vals[vals * sign > 0].abs().sum()

    supp_carriers = sorted(
        set(pos_carriers)
        - set(common_carriers)
        | {
            c
            for c in common_carriers
            if get_total_abs(c, 1)
            >= get_total_abs(c, -1)
        }
    )
    cons_carriers = sorted(
        set(neg_carriers)
        - set(common_carriers)
        | {
            c
            for c in common_carriers
            if get_total_abs(c, 1)
            < get_total_abs(c, -1)
        }
    )

    def get_color(c):
        return label_to_colors.get(
            c,
            carrier_colors.get(
                c,
                "lightgrey",
            ),
        )

    add_legend_patches(
        ax,
        [get_color(c) for c in supp_carriers],
        supp_carriers,
        legend_kw={
            "bbox_to_anchor": (0, -pad),
            "ncol": 1,
            "title": "Supply",
            **legend_kwargs,
        },
    )
    add_legend_patches(
        ax,
        [get_color(c) for c in cons_carriers],
        cons_carriers,
        legend_kw={
            "bbox_to_anchor": (0.5, -pad),
            "ncol": 1,
            "title": "Consumption",
            **legend_kwargs,
        },
    )
    legend_bus_size = settings.get(
        "bus_sizes"
    )
    if legend_bus_size is not None:
        add_legend_semicircles(
            ax,
            [
                s
                * bus_size_factor
                * SEMICIRCLE_CORRECTION_FACTOR
                for s in legend_bus_size
            ],
            [
                f"{s} {settings['unit']}"
                for s in legend_bus_size
            ],
            patch_kw={"color": "#666"},
            legend_kw={
                "bbox_to_anchor": (0, 1),
                **legend_kwargs,
            },
        )
    plt.show()
    return eb

def get_projected_area_factor(ax, boundaries, srid=4326):
    import pyproj
    crs_geographic = pyproj.CRS.from_epsg(srid)
    crs_projected = pyproj.CRS(ax.projection.proj4_init)
    transformer = pyproj.Transformer.from_crs(crs_geographic, crs_projected, always_xy=True)

    lon_c = (boundaries[0] + boundaries[1]) / 2
    lat_c = (boundaries[2] + boundaries[3]) / 2

    px0, py0 = transformer.transform(lon_c, lat_c)
    px1, py1 = transformer.transform(lon_c + 1, lat_c)

    factor = np.sqrt((px1 - px0)**2 + (py1 - py0)**2)
    print(f"lon_c={lon_c}, lat_c={lat_c}, factor={factor}")
    return factor


def plot_methanol_production_map(n, config_plotting, label_to_colors, boundaries):
    weights = n.snapshot_weightings.generators
    locations = n.buses[['location', 'x', 'y']].loc[n.buses['location'] != 'EU'].drop_duplicates(subset='location')
    methanol_links = pd.DataFrame(
        index=n.links[
            ((n.links["bus1"].str.contains("methanol renewable cluster")) |
            (n.links["bus1"].str.contains("methanol pointsource cluster")) |
            (n.links["bus1"].str.contains("EU methanol"))) &
            (~n.links.index.str.contains("methanol renewable cluster") &
            ~n.links.index.str.contains("methanol pointsource cluster"))
        ].index
    )


    for idx in methanol_links.index:
        key = idx[:5] if idx[5] == ' ' else idx[:6]
        methanol_links.loc[idx, 'x'] = locations.loc[key, 'x']
        methanol_links.loc[idx, 'y'] = locations.loc[key, 'y']
        methanol_links.loc[idx, 'location'] = locations.loc[key, 'location']

    for tech in methanol_links.index:
        methanol_links.loc[tech, 'methanol_production'] = abs(
            n.links_t['p1'].loc[:, tech].multiply(weights, axis=0).sum()
        )

    methanol_links.index = methanol_links.index.str.slice(start=7).where(
        methanol_links.index.str[6] == ' ',
        methanol_links.index.str.slice(start=6)
    )

    conversion = config_plotting["plotting"]["balance_map"]['methanol']["unit_conversion"]
    bus_sizes = methanol_links.groupby("location")["methanol_production"].sum().div(conversion)
    print(f'conversion: {conversion}')
    print(f'bus_sizes {bus_sizes}')

    geo_scale = 0.15

    def size_to_radius(size_twh):
        return np.sqrt(abs(size_twh) * 2)

    crs = load_projection(copy.deepcopy(config_plotting["plotting"]))
    print(crs)

    fig, ax = plt.subplots(
        figsize=(5, 6.5),
        subplot_kw={"projection": crs},
        layout="constrained",
    )
    ax.set_extent(boundaries, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor="darkgrey", linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.BORDERS, edgecolor="darkgrey", linewidth=0.3, zorder=1)
    ax.spines['geo'].set_visible(False)

    grouped = methanol_links.groupby("location")
    loc_coords = methanol_links.groupby("location")[["x", "y"]].first()

    area_correction = get_projected_area_factor(ax, boundaries, srid=4326)

    for loc, group in grouped:
        x = float(loc_coords.loc[loc, "x"])
        y = float(loc_coords.loc[loc, "y"])
        total = group["methanol_production"].sum()
        shares = group["methanol_production"] / total
        loc_size = float(bus_sizes.loc[loc]) if loc in bus_sizes.index else 0

        r_corrected = size_to_radius(loc_size) * geo_scale * area_correction
        x_proj, y_proj = ax.projection.transform_point(x, y, ccrs.PlateCarree())

        tech_colors = [label_to_colors.get(tech, "#cccccc") for tech in group.index]

        theta1 = 180
        for share, color in zip(shares, tech_colors):
            dtheta = share * 180
            theta2 = theta1 - dtheta

            angles = np.linspace(np.radians(theta2), np.radians(theta1), 100)
            xs = x_proj + r_corrected * np.cos(angles)
            ys = y_proj + r_corrected * np.sin(angles)
            verts = np.column_stack([xs, ys])
            verts = np.vstack([[x_proj, y_proj], verts, [x_proj, y_proj]])

            poly = Polygon(
                verts,
                closed=True,
                facecolor=color,
                edgecolor="white",
                linewidth=0.3,
                zorder=3,
            )
            ax.add_patch(poly)
            theta1 = theta2

    legend_sizes_twh = [1, 10]
    carrier_unit = config_plotting["plotting"]["balance_map"]['methanol']["unit"]
    add_legend_semicircles(
        ax,
        [s * geo_scale**2 for s in legend_sizes_twh],
        [f"{s} {carrier_unit}" for s in legend_sizes_twh],
        patch_kw={"color": "#666"},
        legend_kw={
            "bbox_to_anchor": (0, 1),
            "labelspacing": 1,
            "loc": "upper left",
            "frameon": False,
            "alignment": "left",
            "title_fontproperties": {"weight": "bold"},
        },
    )

    all_techs = methanol_links.index.unique().tolist()
    tech_colors_list = [label_to_colors.get(t, "#cccccc") for t in all_techs]
    add_legend_patches(
        ax,
        tech_colors_list,
        all_techs,
        legend_kw={
            "bbox_to_anchor": (0, -0.18),
            "ncol": 1,
            "title": "Methanol Production",
            "loc": "upper left",
            "frameon": False,
            "alignment": "left",
            "title_fontproperties": {"weight": "bold"},
        },
    )

    ax.set_title("methanol")
    plt.show()

def plot_FT_production_map(n, config_plotting, label_to_colors, boundaries):
    weights = n.snapshot_weightings.generators
    locations = n.buses[['location', 'x', 'y']].loc[n.buses['location'] != 'EU'].drop_duplicates(subset='location')
    FT_links = pd.DataFrame(
        index=n.links[
            (n.links["bus1"].str.contains("EU oil")) &
            (n.links["carrier"].str.contains("Fischer-Tropsch"))

        ].index
    )

    for idx in FT_links.index:
        key = idx[:5] if idx[5] == ' ' else idx[:6]
        FT_links.loc[idx, 'x'] = locations.loc[key, 'x']
        FT_links.loc[idx, 'y'] = locations.loc[key, 'y']
        FT_links.loc[idx, 'location'] = locations.loc[key, 'location']

    for tech in FT_links.index:
        FT_links.loc[tech, 'FT_production'] = abs(
            n.links_t['p1'].loc[:, tech].multiply(weights, axis=0).sum()
        )

    FT_links.index = FT_links.index.str.slice(start=7).where(
        FT_links.index.str[6] == ' ',
        FT_links.index.str.slice(start=6)
    )

    conversion = config_plotting["plotting"]["balance_map"]['methanol']["unit_conversion"]
    bus_sizes = FT_links.groupby("location")["FT_production"].sum().div(conversion)
    print(f'conversion: {conversion}')
    print(f'bus_sizes {bus_sizes}')

    geo_scale = 0.15

    def size_to_radius(size_twh):
        return np.sqrt(abs(size_twh) * 2)

    crs = load_projection(copy.deepcopy(config_plotting["plotting"]))
    print(crs)

    fig, ax = plt.subplots(
        figsize=(5, 6.5),
        subplot_kw={"projection": crs},
        layout="constrained",
    )
    ax.set_extent(boundaries, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor="darkgrey", linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.BORDERS, edgecolor="darkgrey", linewidth=0.3, zorder=1)
    ax.spines['geo'].set_visible(False)

    grouped = FT_links.groupby("location")
    loc_coords = FT_links.groupby("location")[["x", "y"]].first()

    area_correction = get_projected_area_factor(ax, boundaries, srid=4326)

    for loc, group in grouped:
        x = float(loc_coords.loc[loc, "x"])
        y = float(loc_coords.loc[loc, "y"])
        total = group["FT_production"].sum()
        shares = group["FT_production"] / total
        loc_size = float(bus_sizes.loc[loc]) if loc in bus_sizes.index else 0

        r_corrected = size_to_radius(loc_size) * geo_scale * area_correction
        x_proj, y_proj = ax.projection.transform_point(x, y, ccrs.PlateCarree())

        tech_colors = [label_to_colors.get(tech, "#cccccc") for tech in group.index]

        theta1 = 180
        for share, color in zip(shares, tech_colors):
            dtheta = share * 180
            theta2 = theta1 - dtheta

            angles = np.linspace(np.radians(theta2), np.radians(theta1), 100)
            xs = x_proj + r_corrected * np.cos(angles)
            ys = y_proj + r_corrected * np.sin(angles)
            verts = np.column_stack([xs, ys])
            verts = np.vstack([[x_proj, y_proj], verts, [x_proj, y_proj]])

            poly = Polygon(
                verts,
                closed=True,
                facecolor=color,
                edgecolor="white",
                linewidth=0.3,
                zorder=3,
            )
            ax.add_patch(poly)
            theta1 = theta2

    legend_sizes_twh = [1, 10]
    carrier_unit = config_plotting["plotting"]["balance_map"]['methanol']["unit"]
    add_legend_semicircles(
        ax,
        [s * geo_scale**2 for s in legend_sizes_twh],
        [f"{s} {carrier_unit}" for s in legend_sizes_twh],
        patch_kw={"color": "#666"},
        legend_kw={
            "bbox_to_anchor": (0, 1),
            "labelspacing": 1,
            "loc": "upper left",
            "frameon": False,
            "alignment": "left",
            "title_fontproperties": {"weight": "bold"},
        },
    )

    all_techs = FT_links.index.unique().tolist()
    tech_colors_list = [label_to_colors.get(t, "#cccccc") for t in all_techs]
    add_legend_patches(
        ax,
        tech_colors_list,
        all_techs,
        legend_kw={
            "bbox_to_anchor": (0, -0.18),
            "ncol": 1,
            "title": "Fischer-Tropsch Production",
            "loc": "upper left",
            "frameon": False,
            "alignment": "left",
            "title_fontproperties": {"weight": "bold"},
        },
    )

    ax.set_title("Fischer-Tropsch")
    plt.show()


def calculate_price_of_methanol_cluster(n):
    weights = n.snapshot_weightings.generators
    locations = n.buses.loc[(n.buses['carrier'] == 'AC'), ['location']].location.unique()

    methanol_prices = pd.DataFrame(index=n.snapshots, columns=locations)
    average_price_of_methanol = pd.Series(index=locations, dtype=float)
    co2_intensity_methanol = n.links.loc[n.links.index.str.contains("EU shipping methanol"), 'efficiency2'].iloc[0]
    co2_price = n.global_constraints.loc[n.global_constraints.carrier_attribute.str.contains("co2_emissions"), 'mu'].iloc[0]
    print(f"CO2 price: {co2_price} €/tCO2")

    for node in locations:
        # Methanolisation
        if not n.links.index.str.contains(f"^{node} methanolisation renewable cluster$").any():
            continue

        if n.links.loc[n.links.index == f"{node} methanolisation renewable cluster", 'p_nom_opt'].iloc[0] > 1:
            price_of_methanol = n.buses_t.marginal_price.loc[:, node + ' methanol renewable cluster']
            production_of_methanol = n.links_t.p1.loc[:, f"{node} methanolisation renewable cluster"].abs().multiply(weights, axis=0)

            price_of_methanol = price_of_methanol - co2_intensity_methanol * co2_price
            average = (production_of_methanol * price_of_methanol).sum() / production_of_methanol.sum()

            methanol_prices[node] = price_of_methanol
            average_price_of_methanol[node] = average

    return methanol_prices, average_price_of_methanol


def plot_load_duration_curve(n,tech_dictionary):

    h_per_snapshot = n.snapshot_weightings.generators.iloc[0]

    #for node in ["GB2 06", "GB2 13", "DK0 0"]:
    for node in ["PT0 2"]:

        print(f"Processing node: {node}")  # Debug statement to track which node is being processed

        load_durations=pd.DataFrame()
        pnoms={}

        if not any((node + " ")in techname for _, techname in tech_dictionary):
            continue

        for techtype, techname in tech_dictionary:
            
            timeserie=None
            pnom=0
        
            if techtype=="generator"  and ((node + " ")in techname):
                timeserie=(n.generators_t["p"].loc[:, techname])
                pnom=n.generators.loc[techname, "p_nom_opt"]
                print(f"{techname} p_nom: {pnom} MW")

                
            elif techtype=="link" and ((node + " ") in techname and "methanolisation" not in techname):
                timeserie=(n.links_t["p0"].loc[:, techname]).abs()
                pnom = (n.links.loc[techname,"p_nom_opt"])
                print(f"{techname} p_nom: {pnom} MW")

            elif techtype=="link" and ((node + " ") in techname and "methanolisation" in techname):
                timeserie=(n.links_t["p2"].loc[:, techname]).abs()
                pnom = abs((n.links.loc[techname,"p_nom_opt"])*((n.links.loc[techname,"efficiency2"])))
                print(f"{techname} p_nom: {pnom} MW")





            
 
            load_duration=timeserie

            if load_duration is None:
                continue
            
            techname = techname[len(node):].lstrip()
            if techtype == "generator":
                techname = techname[2:]

            load_duration=load_duration.sort_values(ascending=False).reset_index(drop=True)

            
            
            if not load_duration.le(0.0001).all():
                load_durations[techname] = load_duration
                pnoms[techname] = pnom 



        load_durations.index=load_durations.index*(h_per_snapshot)# convert to hours

        print(load_durations.dtypes)
        print(load_durations.head())


        fig,ax=plt.subplots(figsize=(10,5))
        load_durations.plot(ax=ax,
                        ylabel='Power [MW]',
                        xlabel= 'hours',
                        title=f"Load Duration Curves for renewable technologies in {node}",
                        color=[label_to_colors.get(label, "#cccccc")  # fallback color
                               for label in load_durations.columns]
                        )
    
        ax.set_xlim(left=0)
        ax.set_xlim(right=8760)
        ax.set_ylim(bottom=0)
        ax.set_ylim(top=max(pnoms.values())*1.1)

        tech_list = list(load_durations.columns)
        lines = ax.get_lines()

        for i, techname in enumerate(tech_list):
            color = lines[i].get_color()        # same color as load_duration curve
                                
            # horizontal line
            ax.hlines(y=pnoms[techname],
                    xmin=0,
                    xmax=8760,
                    colors=color,
                    linestyles="dashed")
            print (f"{techname} p_nom: {pnoms[techname]} MW")
            y_line = pnoms[techname]          
            y_offset = 0.003                 

            # label next to the line
            ax.text(
            8000,
            y_line - y_offset,
            "p_nom",
            color=color,
            va='top',
            fontsize=10,                 
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=color,
                alpha=0.8
            )
        )

        
        leg = ax.legend(
            loc='upper left',           
            bbox_to_anchor=(1.02,1),
            frameon=True,
        )

        
        fig.subplots_adjust(bottom=0.25)   
        

        
        plt.show()

    return load_durations, pnoms




#%%
#Iberian Peninsula


networks_folder=r"results/Iberic40_2035_industrial_clusters_both/all/networks"
config_plotting = yaml.safe_load(Path("config/plotting.default.yaml").read_text())                        
regions=gdp.read_file(r'resources/Iberic40_2035_industrial_clusters/all/regions_onshore_base_s_40.geojson').set_index("name")

#%%
records = []
for path in sorted(Path(networks_folder).glob("*.nc")):
    wc = parse_wildcards(path)
    records.append({**wc, "path": path})

df = pd.DataFrame(records)

df = add_co2_captured(df)
df = add_co2_marginal_price(df)
df = add_renewable_capacity_cluster(df)

#%%

fig, ax = plot_heatmap_buycap_vs_cr(df, 0, 0, "co2 captured cluster [Mtons/year]", "Purples", cr_range=None, buycap_range=(0, 1), title="CO2 captured in renewable clusters [Mtons/year] - Iberian Peninsula 2035")
fig, ax = plot_heatmap_sellcap_vs_cr(df, 0, 0, "co2 captured cluster [Mtons/year]", "Purples", cr_range=None, sellcap_range=(0, 0.4), title="CO2 captured in renewable clusters [Mtons/year] - Iberian Peninsula 2035")
fig, ax = plot_heatmap_bothcap_vs_cr(df, 0, 0, "co2 captured cluster [Mtons/year]", "Purples", cr_range=None, bothcap_range=(0, 0.4), title="CO2 captured in renewable clusters [Mtons/year] - Iberian Peninsula 2035")

fig, ax = plot_heatmap_buycap_vs_cr(df, 0, 0, "co2 price [euro/ton]", "Reds", cr_range=None, buycap_range=(0, 1), title="CO2 marginal price [euro/ton] - Iberian Peninsula 2035")
fig, ax = plot_heatmap_sellcap_vs_cr(df, 0, 0, "co2 price [euro/ton]", "Reds", cr_range=None, sellcap_range=(0, 0.4), title="CO2 marginal price [euro/ton] - Iberian Peninsula 2035")
fig, ax = plot_heatmap_bothcap_vs_cr(df, 0, 0, "co2 price [euro/ton]", "Reds", cr_range=None, bothcap_range=(0, 0.4), title="CO2 marginal price [euro/ton] - Iberian Peninsula 2035")

fig, ax = plot_heatmap_buycap_vs_cr(df, 0, 0, "renewable capacity cluster [MW]", "Greens", cr_range=None, buycap_range=(0, 1), title="Renewable capacity in clusters [MW] - Iberian Peninsula 2035")
fig, ax = plot_heatmap_sellcap_vs_cr(df, 0, 0, "renewable capacity cluster [MW]", "Greens", cr_range=None, sellcap_range=(0, 0.4), title="Renewable capacity in clusters [MW] - Iberian Peninsula 2035")
fig, ax = plot_heatmap_bothcap_vs_cr(df, 0, 0, "renewable capacity cluster [MW]", "Greens", cr_range=None, bothcap_range=(0, 0.4), title="Renewable capacity in clusters [MW] - Iberian Peninsula 2035")

#%%
n = pypsa.Network(str(df.loc[(df.CR == 0.2) & (df.BUYcap == 0) & (df.SELLcap == 0.0)  & (df.BOTHcap == 0.1), "path"].iloc[0]))

cluster_components = (
    [("link", idx) for idx in n.links.index if ("cluster" in idx and "charger" not in idx and "methanol renewable cluster" not in idx and n.links.loc[idx, "p_nom_opt"] > 0.1)]
    + [("generator", idx) for idx in n.generators.index if ("cluster" in idx and n.generators.loc[idx, "p_nom_opt"] > 0.1)]
)
load_durations, pnoms = plot_load_duration_curve(n, cluster_components)

#%%
# n = pypsa.Network(str(df.loc[(df.CR == 0) & (df.BUYcap == 0) & (df.SELLcap == 0), "path"].iloc[0]))
# eb_iberian=plot_balance_map_methanol_price(n, regions, config_plotting, label_to_colors, boundaries=[-15, 12, 35, 48])
# #%%
n = pypsa.Network(str(df.loc[(df.CR == 0.2) & (df.BUYcap == 0) & (df.SELLcap == 0.0)  & (df.BOTHcap == 0.1), "path"].iloc[0]))
plot_methanol_production_map(n, config_plotting, label_to_colors, boundaries=[-15, 12, 35, 48])
plot_FT_production_map(n, config_plotting, label_to_colors, boundaries=[-15, 12, 35, 48])

methanol_prices_cluster, average_price_of_methanol_cluster = calculate_price_of_methanol_cluster(n)

# %%
#Noridc Countries


networks_folder=r"results/Noridcs100_2035_industrial_clusters/all/networks"
config_plotting = yaml.safe_load(Path("config/plotting.default.yaml").read_text())                        
regions=gdp.read_file(r'resources/Noridcs100_2035_industrial_clusters/all/regions_onshore_base_s_100.geojson').set_index("name")


records = []
for path in sorted(Path(networks_folder).glob("*.nc")):
    wc = parse_wildcards(path)
    records.append({**wc, "path": path})

df = pd.DataFrame(records)

df = add_co2_captured(df)
df = add_co2_marginal_price(df)
df = add_renewable_capacity_cluster(df)

#%%
fig, ax = plot_heatmap_buycap_vs_cr(df, 0, 0, "co2 captured cluster [Mtons/year]", "Purples", cr_range=None, buycap_range=(0, 1), title="CO2 captured in renewable clusters [Mtons/year] - Nordics 2035")
fig, ax = plot_heatmap_sellcap_vs_cr(df, 0, 0, "co2 captured cluster [Mtons/year]", "Purples", cr_range=None, sellcap_range=(0, 0.4), title="CO2 captured in renewable clusters [Mtons/year] - Nordics 2035")

fig, ax = plot_heatmap_buycap_vs_cr(df, 0, 0, "co2 price [euro/ton]", "Reds", cr_range=None, buycap_range=(0, 1), title="CO2 marginal price [euro/ton] - Nordics 2035")
fig, ax = plot_heatmap_sellcap_vs_cr(df, 0, 0, "co2 price [euro/ton]", "Reds", cr_range=None, sellcap_range=(0, 0.4), title="CO2 marginal price [euro/ton] - Nordics 2035")

fig, ax = plot_heatmap_buycap_vs_cr(df, 0, 0, "renewable capacity cluster [MW]", "Greens", cr_range=None, buycap_range=(0, 1), title="Renewable capacity in clusters [MW] - Nordics 2035")
fig, ax = plot_heatmap_sellcap_vs_cr(df, 0, 0, "renewable capacity cluster [MW]", "Greens", cr_range=None, sellcap_range=(0, 0.4), title="Renewable capacity in clusters [MW] - Nordics 2035")

#%%
n = pypsa.Network(str(df.loc[(df.CR == 0) & (df.BUYcap == 0) & (df.SELLcap == 0.2), "path"].iloc[0]))

cluster_components = (
    [("link", idx) for idx in n.links.index if ("cluster" in idx and "charger" not in idx and "methanol renewable cluster" not in idx and n.links.loc[idx, "p_nom_opt"] > 0.1)]
    + [("generator", idx) for idx in n.generators.index if ("cluster" in idx and n.generators.loc[idx, "p_nom_opt"] > 0.1)]
)
load_durations, pnoms = plot_load_duration_curve(n, cluster_components)
# n = pypsa.Network(str(df.loc[(df.CR == 0) & (df.BUYcap == 0) & (df.SELLcap == 0), "path"].iloc[0]))
# eb_nordics=plot_balance_map_methanol_price(n, regions, config_plotting, label_to_colors, boundaries=[-10, 28, 46, 73])

#%%
n = pypsa.Network(str(df.loc[(df.CR == 0.0) & (df.BUYcap == 0.25) & (df.SELLcap == 0), "path"].iloc[0]))
plot_methanol_production_map(n, config_plotting, label_to_colors, boundaries=[-10, 28, 46, 73])
plot_FT_production_map(n, config_plotting, label_to_colors, boundaries=[-10, 28, 46, 73])
methanol_prices_cluster, average_price_of_methanol_cluster = calculate_price_of_methanol_cluster(n)

#%%
methanol_prices_cluster, average_price_of_methanol_cluster = calculate_price_of_methanol_cluster(n)
n = pypsa.Network(str(df.loc[(df.CR == 0.5) & (df.BUYcap == 0) & (df.SELLcap == 0.1), "path"].iloc[0]))
plot_methanol_production_map(n, config_plotting, label_to_colors, boundaries=[-10, 28, 46, 73])
# %%
