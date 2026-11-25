#%%
import copy
import pypsa
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
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
import matplotlib.dates as mdates
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
    'electricity renewable cluster both': "#6e6e6e",

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
    'grid connection (buy) renewable cluster':'#858ba8',
    'grid connection (both) renewable cluster': "#6e6e6e",

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


def parse_wildcards(path: Path) -> dict:
    name = path.name
    return {
        "CR": find_wildcard_value(name, "CR"),
        "BUYcap": find_wildcard_value(name, "BUYcap"),
        "SELLcap": find_wildcard_value(name, "SELLcap"),
        "BOTHcap": find_wildcard_value(name, "BOTHcap"),
    }

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
    # print(f"co2_intensity_FT: {co2_intensity_FT}")

    co2_intensity_methanol = n.links.loc[
        n.links.index.str.contains("EU shipping methanol"), 'efficiency2'
    ].iloc[0]
    # print(f"co2_intensity_methanol: {co2_intensity_methanol}")

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

def compute_ft_and_methanol_prices_renewable_cluster(n):
    """
    Compute FT oil and methanol prices per node for production happening
    inside the renewable cluster, based on marginal costs of inputs
    (H2, CO2, electricity, heat) and allocated capital costs.

    Identical in logic to compute_ft_and_methanol_prices, but pulls
    links and buses from the 'renewable cluster' variants (i.e. names
    with ' renewable cluster' appended).

    Parameters
    ----------
    n : pypsa.Network
        The solved network (generic name, e.g. n_base).

    Returns
    -------
    FT_oil_prices_rc : pd.DataFrame
    methanol_prices_rc : pd.DataFrame
    FT_oil_prices_average_rc : pd.Series (indexed by location)
    methanol_prices_average_rc : pd.Series (indexed by location)
    methanol_cost_shares_rc : pd.DataFrame (indexed by location)
        Per-unit contribution of each cost component to the average
        methanol price (columns: 'H2_share', 'O_and_M_share',
        'capital_share', 'electricity_share', 'CO2_share', 'heat_share').
        Each is that component's term from the price_of_methanol formula,
        summed over snapshots and divided by total methanol production,
        computed the same way as methanol_prices_average_rc so the six
        shares sum to it exactly. 'CO2_share' additionally nets out the
        CO2 allowance credit (-co2_intensity_methanol * co2_price), and
        'heat_share' is negative since it's a credit in the price formula.
    """

    weights = n.snapshot_weightings.generators

    locations = n.buses.loc[(n.buses['carrier'] == 'AC'), ['location']].location.unique()

    methanol_prices_rc = pd.DataFrame(index=n.snapshots, columns=locations)
    FT_oil_prices_rc = pd.DataFrame(index=n.snapshots, columns=locations)
    capital_cost_allocated_FT_rc = pd.DataFrame(index=n.snapshots, columns=locations)
    capital_cost_allocated_methanolisation_rc = pd.DataFrame(index=n.snapshots, columns=locations)

    FT_oil_prices_average_rc = pd.Series(index=locations, dtype=float)
    methanol_prices_average_rc = pd.Series(index=locations, dtype=float)

    methanol_cost_shares_rc = pd.DataFrame(
        index=locations,
        columns=['H2_share', 'O_and_M_share', 'capital_share', 'electricity_share', 'CO2_share', 'heat_share'],
        dtype=float,
    )

    co2_intensity_FT = (
        n.links.loc[
            n.links.index.str.contains("Fischer-Tropsch") & n.links.index.str.contains("renewable cluster"),
            'efficiency2'
        ].iloc[0]
        / n.links.loc[
            n.links.index.str.contains("Fischer-Tropsch") & n.links.index.str.contains("renewable cluster"),
            'efficiency'
        ].iloc[0]
    )

    co2_intensity_methanol = n.links.loc[
        n.links.index.str.contains("EU shipping methanol"),
        'efficiency2'
    ].iloc[0]

    co2_price = n.global_constraints.loc[
        n.global_constraints.carrier_attribute.str.contains("co2_emissions"), 'mu'
    ].iloc[0]

    for node in locations:

        # ---------------- Fischer-Tropsch (renewable cluster) ----------------
        link_FT = f"{node} Fischer-Tropsch renewable cluster"
        if n.links.index.str.contains(f"^{link_FT}$").any():

            if n.links.loc[n.links.index == link_FT, 'p_nom_opt'].iloc[0] > 1:
                O_and_M_cost = n.links.loc[n.links.index == link_FT, 'marginal_cost'].iloc[0]
                capital_cost = (
                    n.links.loc[n.links.index == link_FT, 'capital_cost'].iloc[0]
                    * n.links.loc[n.links.index == link_FT, 'p_nom_opt'].iloc[0]
                )

                price_of_H2 = n.buses_t.marginal_price.loc[:, node + ' H2 renewable cluster']
                consumption_of_H2 = n.links_t.p0.loc[:, link_FT]
                consumption_of_H2_total = n.links_t.p0.loc[:, link_FT].multiply(weights, axis=0).sum()

                capital_cost_allocated_FT_rc[node] = capital_cost * consumption_of_H2 / consumption_of_H2_total

                price_of_CO2 = n.buses_t.marginal_price.loc[:, node + ' co2 stored renewable cluster']
                consumption_of_CO2 = n.links_t.p2.loc[:, link_FT]

                price_of_heat = 0
                production_of_heat = 0
                if node + ' urban central heat renewable cluster' in n.buses_t.marginal_price.columns:
                    price_of_heat = n.buses_t.marginal_price.loc[:, node + ' urban central heat renewable cluster']
                    production_of_heat = n.links_t.p3.loc[:, link_FT].abs()

                production_of_FT_oil = n.links_t.p1.loc[:, link_FT].abs()

                price_of_FT_oil = (
                    price_of_H2 * consumption_of_H2
                    + O_and_M_cost * consumption_of_H2
                    + capital_cost_allocated_FT_rc[node]
                    + price_of_CO2 * consumption_of_CO2
                    - price_of_heat * production_of_heat
                ) / production_of_FT_oil - (-co2_intensity_FT * co2_price)

                FT_oil_prices_rc[node] = price_of_FT_oil

                FT_oil_prices_average_rc[node] = (
                    price_of_H2 * consumption_of_H2
                    + O_and_M_cost * consumption_of_H2
                    + capital_cost_allocated_FT_rc[node]
                    + price_of_CO2 * consumption_of_CO2
                    - price_of_heat * production_of_heat
                ).sum() / production_of_FT_oil.sum() - (-co2_intensity_FT * co2_price)

        # ---------------- Methanolisation (renewable cluster) ----------------
        link_meth = f"{node} methanolisation renewable cluster"
        if n.links.index.str.contains(f"^{link_meth}$").any():

            if n.links.loc[n.links.index == link_meth, 'p_nom_opt'].iloc[0] > 1:
                O_and_M_cost = n.links.loc[n.links.index == link_meth, 'marginal_cost'].iloc[0]
                capital_cost = (
                    n.links.loc[n.links.index == link_meth, 'capital_cost'].iloc[0]
                    * n.links.loc[n.links.index == link_meth, 'p_nom_opt'].iloc[0]
                )

                price_of_H2 = n.buses_t.marginal_price.loc[:, node + ' H2 renewable cluster']
                consumption_of_H2 = n.links_t.p0.loc[:, link_meth]
                consumption_of_H2_total = n.links_t.p0.loc[:, link_meth].multiply(weights, axis=0).sum()

                capital_cost_allocated_methanolisation_rc[node] = (
                    capital_cost * consumption_of_H2 / consumption_of_H2_total
                )

                price_of_CO2 = n.buses_t.marginal_price.loc[:, node + ' co2 stored renewable cluster']
                consumption_of_CO2 = n.links_t.p3.loc[:, link_meth]
                price_of_electricity = n.buses_t.marginal_price.loc[:, node + ' renewable cluster']
                consumption_of_electricity = n.links_t.p2.loc[:, link_meth]

                price_of_heat = 0
                production_of_heat = 0
                if node + ' urban central heat renewable cluster' in n.buses_t.marginal_price.columns:
                    price_of_heat = n.buses_t.marginal_price.loc[:, node + ' urban central heat renewable cluster']
                    production_of_heat = n.links_t.p4.loc[:, link_meth].abs()

                production_of_methanol = n.links_t.p1.loc[:, link_meth].abs()

                price_of_methanol = (
                    price_of_H2 * consumption_of_H2
                    + O_and_M_cost * consumption_of_H2
                    + capital_cost_allocated_methanolisation_rc[node]
                    + price_of_electricity * consumption_of_electricity
                    + price_of_CO2 * consumption_of_CO2
                    - price_of_heat * production_of_heat
                ) / production_of_methanol - co2_intensity_methanol * co2_price

                methanol_prices_rc[node] = price_of_methanol

                methanol_prices_average_rc[node] = (
                    price_of_H2 * consumption_of_H2
                    + O_and_M_cost * consumption_of_H2
                    + capital_cost_allocated_methanolisation_rc[node]
                    + price_of_electricity * consumption_of_electricity
                    + price_of_CO2 * consumption_of_CO2
                    - price_of_heat * production_of_heat
                ).sum() / production_of_methanol.sum() - co2_intensity_methanol * co2_price

                # ---- shares of the average methanol price, per cost component ----
                methanol_cost_shares_rc.loc[node, 'H2_share'] = (
                    (price_of_H2 * consumption_of_H2).sum() / production_of_methanol.sum()
                )
                methanol_cost_shares_rc.loc[node, 'O_and_M_share'] = (
                    (O_and_M_cost * consumption_of_H2).sum() / production_of_methanol.sum()
                )
                methanol_cost_shares_rc.loc[node, 'capital_share'] = (
                    capital_cost_allocated_methanolisation_rc[node].sum() / production_of_methanol.sum()
                )
                methanol_cost_shares_rc.loc[node, 'electricity_share'] = (
                    (price_of_electricity * consumption_of_electricity).sum() / production_of_methanol.sum()
                )
                methanol_cost_shares_rc.loc[node, 'CO2_share'] = (
                    (price_of_CO2 * consumption_of_CO2).sum() / production_of_methanol.sum()
                    - co2_intensity_methanol * co2_price
                )
                methanol_cost_shares_rc.loc[node, 'heat_share'] = (
                    -(price_of_heat * production_of_heat).sum() / production_of_methanol.sum()
                    if isinstance(production_of_heat, pd.Series)
                    else 0.0
                )

    return (
        FT_oil_prices_rc,
        methanol_prices_rc,
        FT_oil_prices_average_rc,
        methanol_prices_average_rc,
        methanol_cost_shares_rc,
    )
def compute_avg_electricity_price(n):
    """
    Compute the load-weighted average electricity price per node.

    avg_electricity_price = sum_t(price_t * el_t) / sum_t(el_t)

    Parameters
    ----------
    n : pypsa.Network
        The solved network.

    Returns
    -------
    avg_electricity_price : pd.Series (indexed by location)
    """

    locations = n.buses.loc[(n.buses['carrier'] == 'AC'), ['location']].location.unique()

    avg_electricity_price = pd.Series(index=locations, dtype=float)

    for node in locations:
        price_t = n.buses_t.marginal_price.loc[:, node]
        load_cols = n.loads_t.p.columns[n.loads_t.p.columns.str.contains(node)]
        el_t = n.loads_t.p.loc[:, load_cols].sum(axis=1)
        avg_electricity_price[node] = (price_t * el_t).sum() / el_t.sum()

    return avg_electricity_price


def compute_avg_capacity_factor(n, carrier):
    """
    Compute the time-averaged capacity factor per node for a given
    generator carrier (e.g. "onwind" or "solar").

    Each node has at most one generator whose index is exactly
    "{node} {carrier}"; nodes without such a generator are left as NaN.

    Parameters
    ----------
    n : pypsa.Network
        The solved network.
    carrier : str
        Generator carrier suffix, e.g. "onwind" or "solar".

    Returns
    -------
    avg_capacity_factor : pd.Series (indexed by location)
    """

    locations = n.buses.loc[(n.buses['carrier'] == 'AC'), ['location']].location.unique()
    print(locations)

    avg_capacity_factor = pd.Series(index=locations, dtype=float)

    for node in locations:
        gen_name = f"{node} 0 {carrier}"
        if gen_name in n.generators_t.p_max_pu.columns:
            avg_capacity_factor[node] = n.generators_t.p_max_pu[gen_name].mean()

    return avg_capacity_factor


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
    # print(f"lon_c={lon_c}, lat_c={lat_c}, factor={factor}")
    return factor


def plot_methanol_production_el_price_colormap_map(n, regions, config_plotting, label_to_colors, boundaries):
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
    # print(f'conversion: {conversion}')
    # print(f'bus_sizes {bus_sizes}')

    geo_scale = 0.15

    def size_to_radius(size_twh):
        return np.sqrt(abs(size_twh) * 2)

    # --------------------------------------------------
    # AVERAGE ELECTRICITY PRICE (per node, load-weighted)
    # --------------------------------------------------
    avg_electricity_price = compute_avg_electricity_price(n)
    regions = regions.copy()
    # NOTE: no fillna here - nodes with no price stay NaN
    # so geopandas skips them instead of drawing them as 0.
    regions["electricity_price"] = avg_electricity_price.reindex(regions.index)

    vmin = regions.electricity_price.min()  # NaNs ignored automatically
    vmax = regions.electricity_price.max()
    if vmin == vmax:
        vmax += 1

    red_cmap = mcolors.LinearSegmentedColormap.from_list(
        "red_scale",
        ["#ffe5e5", "#b30000", "#4d0000"],
    )

    crs = load_projection(copy.deepcopy(config_plotting["plotting"]))
    # print(crs)

    fig, ax = plt.subplots(
        figsize=(5, 6.5),
        subplot_kw={"projection": crs},
        layout="constrained",
    )
    ax.set_extent(boundaries, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
    regions.to_crs(crs.proj4_init).plot(
        ax=ax,
        column="electricity_price",
        cmap=red_cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolor="None",
        linewidth=0,
        zorder=0.5,
        missing_kwds={
            "color": "none",
        },
    )
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
        if loc_size <= 0.1:
            continue

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

    ax.set_title("Methanol Production", fontsize=8)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=red_cmap, norm=norm)
    cbar = fig.colorbar(
        sm,
        ax=ax,
        label="Average electricity price [€/MWh]",
        shrink=0.95,
        pad=0.0,
        aspect=50,
        orientation="horizontal",
    )
    from matplotlib.ticker import FormatStrFormatter

    cbar.outline.set_edgecolor("None")
    cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cbar.set_label("Average electricity price [€/MWh]", fontsize=6)
    cbar.ax.tick_params(labelsize=6)

    # Legend text/handle sizing to match the co2-stored map: compact
    # patches for the tech legend, default handle sizing for the
    # semicircle-size legend (reusing the compact handlelength/
    # handleheight there distorts the relative circle sizes).
    legend_kwargs = {
        "loc": "upper left",
        "frameon": False,
        "alignment": "left",
        "fontsize": 6,
        "title_fontproperties": {
            "weight": "bold",
            "size": 7,
        },
        "handlelength": 1.0,
        "handleheight": 1.0,
        "handletextpad": 0.4,
        "labelspacing": 0.25,
        "borderaxespad": 0.2,
    }

    size_legend_kwargs = {
        "loc": "upper left",
        "frameon": False,
        "alignment": "left",
        "fontsize": 6,
        "title_fontproperties": {
            "weight": "bold",
            "size": 7,
        },
        "labelspacing": 0.4,
    }

    all_techs = methanol_links.index.unique().tolist()
    tech_colors_list = [label_to_colors.get(t, "#cccccc") for t in all_techs]
    add_legend_patches(
        ax,
        tech_colors_list,
        all_techs,
        legend_kw={
            "bbox_to_anchor": (0, -0.10),
            "ncol": 1,
            "title": "Methanol Production",
            **legend_kwargs,
        },
    )

    legend_sizes_twh = [1, 10]
    carrier_unit = config_plotting["plotting"]["balance_map"]['methanol']["unit"]
    add_legend_semicircles(
        ax,
        [s * geo_scale**2 for s in legend_sizes_twh],
        [f"{s} {carrier_unit}" for s in legend_sizes_twh],
        patch_kw={"color": "#666"},
        legend_kw={
            "bbox_to_anchor": (0, 1),
            **size_legend_kwargs,
        },
    )

    plt.show()

def plot_methanol_production_onwind_cf_colormap(n, regions, config_plotting, label_to_colors, boundaries):
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
    # print(f'conversion: {conversion}')
    # print(f'bus_sizes {bus_sizes}')

    geo_scale = 0.15

    def size_to_radius(size_twh):
        return np.sqrt(abs(size_twh) * 2)

    # --------------------------------------------------
    # AVERAGE ONSHORE WIND CAPACITY FACTOR (per node)
    # --------------------------------------------------
    avg_onwind_cf = compute_avg_capacity_factor(n, "onwind")
    regions = regions.copy()
    # NOTE: no fillna here - nodes with no onwind generator stay NaN
    # so geopandas skips them instead of drawing them as 0.
    regions["onwind_cf"] = avg_onwind_cf.reindex(regions.index)

    vmin = regions.onwind_cf.min()  # NaNs ignored automatically
    vmax = regions.onwind_cf.max()
    if vmin == vmax:
        vmax += 1

    blue_cmap = mcolors.LinearSegmentedColormap.from_list(
        "blue_scale",
        ["#e5f0ff", "#0d47a1", "#00204d"],
    )

    crs = load_projection(copy.deepcopy(config_plotting["plotting"]))
    # print(crs)

    fig, ax = plt.subplots(
        figsize=(5, 6.5),
        subplot_kw={"projection": crs},
        layout="constrained",
    )
    ax.set_extent(boundaries, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
    regions.to_crs(crs.proj4_init).plot(
        ax=ax,
        column="onwind_cf",
        cmap=blue_cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolor="None",
        linewidth=0,
        zorder=0.5,
        missing_kwds={
            "color": "none",
        },
    )
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
        if loc_size <= 0.1:
            continue

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

    ax.set_title("Methanol Production", fontsize=10)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=blue_cmap, norm=norm)
    cbar = fig.colorbar(
        sm,
        ax=ax,
        label="Average onshore wind capacity factor [-]",
        shrink=0.95,
        pad=0.0,
        aspect=50,
        orientation="horizontal",
    )
    from matplotlib.ticker import FormatStrFormatter

    cbar.outline.set_edgecolor("None")
    cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar.set_label("Average onshore wind capacity factor [-]", fontsize=9)
    cbar.ax.tick_params(labelsize=6)

    legend_kwargs = {
        "loc": "upper left",
        "frameon": False,
        "alignment": "left",
        "fontsize": 8,
        "title_fontproperties": {
            "weight": "bold",
            "size": 9,
        },
        "handlelength": 1.0,
        "handleheight": 1.0,
        "handletextpad": 0.4,
        "labelspacing": 0.25,
        "borderaxespad": 0.2,
    }

    size_legend_kwargs = {
        "loc": "upper left",
        "frameon": False,
        "alignment": "left",
        "fontsize": 8,
        "title_fontproperties": {
            "weight": "bold",
            "size": 9,
        },
        "labelspacing": 0.4,
    }

    all_techs = methanol_links.index.unique().tolist()
    tech_colors_list = [label_to_colors.get(t, "#cccccc") for t in all_techs]
    add_legend_patches(
        ax,
        tech_colors_list,
        all_techs,
        legend_kw={
            "bbox_to_anchor": (0, -0.10),
            "ncol": 1,
            "title": "Methanol Production",
            **legend_kwargs,
        },
    )

    legend_sizes_twh = [1, 10]
    carrier_unit = config_plotting["plotting"]["balance_map"]['methanol']["unit"]
    add_legend_semicircles(
        ax,
        [s * geo_scale**2 for s in legend_sizes_twh],
        [f"{s} {carrier_unit}" for s in legend_sizes_twh],
        patch_kw={"color": "#666"},
        legend_kw={
            "bbox_to_anchor": (0, 1),
            **size_legend_kwargs,
        },
    )

    plt.show()

def plot_methanol_production_solar_cf_colormap(n, regions, config_plotting, label_to_colors, boundaries):
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
    # print(f'conversion: {conversion}')
    # print(f'bus_sizes {bus_sizes}')

    geo_scale = 0.15

    def size_to_radius(size_twh):
        return np.sqrt(abs(size_twh) * 2)

    # --------------------------------------------------
    # AVERAGE SOLAR CAPACITY FACTOR (per node)
    # --------------------------------------------------
    avg_solar_cf = compute_avg_capacity_factor(n, "solar")
    regions = regions.copy()
    # NOTE: no fillna here - nodes with no solar generator stay NaN
    # so geopandas skips them instead of drawing them as 0.
    regions["solar_cf"] = avg_solar_cf.reindex(regions.index)

    vmin = regions.solar_cf.min()  # NaNs ignored automatically
    vmax = regions.solar_cf.max()
    if vmin == vmax:
        vmax += 1

    orange_cmap = mcolors.LinearSegmentedColormap.from_list(
        "orange_scale",
        ["#fff2e0", "#e07b00", "#663500"],
    )

    crs = load_projection(copy.deepcopy(config_plotting["plotting"]))
    # print(crs)

    fig, ax = plt.subplots(
        figsize=(5, 6.5),
        subplot_kw={"projection": crs},
        layout="constrained",
    )
    ax.set_extent(boundaries, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
    regions.to_crs(crs.proj4_init).plot(
        ax=ax,
        column="solar_cf",
        cmap=orange_cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolor="None",
        linewidth=0,
        zorder=0.5,
        missing_kwds={
            "color": "none",
        },
    )
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
        if loc_size <= 0.1:
            continue

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

    ax.set_title("Methanol Production", fontsize=8)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=orange_cmap, norm=norm)
    cbar = fig.colorbar(
        sm,
        ax=ax,
        label="Average solar capacity factor [-]",
        shrink=0.95,
        pad=0.0,
        aspect=50,
        orientation="horizontal",
    )
    from matplotlib.ticker import FormatStrFormatter

    cbar.outline.set_edgecolor("None")
    cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar.set_label("Average solar capacity factor [-]", fontsize=6)
    cbar.ax.tick_params(labelsize=6)

    legend_kwargs = {
        "loc": "upper left",
        "frameon": False,
        "alignment": "left",
        "fontsize": 6,
        "title_fontproperties": {
            "weight": "bold",
            "size": 7,
        },
        "handlelength": 1.0,
        "handleheight": 1.0,
        "handletextpad": 0.4,
        "labelspacing": 0.25,
        "borderaxespad": 0.2,
    }

    size_legend_kwargs = {
        "loc": "upper left",
        "frameon": False,
        "alignment": "left",
        "fontsize": 6,
        "title_fontproperties": {
            "weight": "bold",
            "size": 7,
        },
        "labelspacing": 0.4,
    }

    all_techs = methanol_links.index.unique().tolist()
    tech_colors_list = [label_to_colors.get(t, "#cccccc") for t in all_techs]
    add_legend_patches(
        ax,
        tech_colors_list,
        all_techs,
        legend_kw={
            "bbox_to_anchor": (0, -0.10),
            "ncol": 1,
            "title": "Methanol Production",
            **legend_kwargs,
        },
    )

    legend_sizes_twh = [1, 10]
    carrier_unit = config_plotting["plotting"]["balance_map"]['methanol']["unit"]
    add_legend_semicircles(
        ax,
        [s * geo_scale**2 for s in legend_sizes_twh],
        [f"{s} {carrier_unit}" for s in legend_sizes_twh],
        patch_kw={"color": "#666"},
        legend_kw={
            "bbox_to_anchor": (0, 1),
            **size_legend_kwargs,
        },
    )

    plt.show()




def plot_dac_captured_co2_onwind_cf_colormap(n, regions, config_plotting, label_to_colors, boundaries):
    weights = n.snapshot_weightings.generators
    locations = n.buses[['location', 'x', 'y']].loc[n.buses['location'] != 'EU'].drop_duplicates(subset='location')

    # DAC links: identified purely by 'DAC' appearing in the link name
    dac_links = pd.DataFrame(
        index=n.links[n.links.index.str.contains("DAC")].index
    )

    for idx in dac_links.index:
        key = idx[:5] if idx[5] == ' ' else idx[:6]
        dac_links.loc[idx, 'x'] = locations.loc[key, 'x']
        dac_links.loc[idx, 'y'] = locations.loc[key, 'y']
        dac_links.loc[idx, 'location'] = locations.loc[key, 'location']

    # Captured CO2 comes out of p2 (the CO2 bus port), not p1
    for tech in dac_links.index:
        dac_links.loc[tech, 'co2_captured'] = abs(
            n.links_t['p2'].loc[:, tech].multiply(weights, axis=0).sum()
        )

    # Strip the location prefix (e.g. "DE0 0 ") to leave just the tech label
    tech_raw = dac_links.index.str.slice(start=7).where(
        dac_links.index.str[6] == ' ',
        dac_links.index.str.slice(start=6)
    )

    # Collapse everything down to just two buckets: any 'renewable cluster'
    # variant, and everything else (urban central / urban decentral / ...)
    # merged into a plain 'DAC' bucket.
    dac_links['tech'] = np.where(
        tech_raw.str.contains('renewable cluster'),
        'DAC renewable cluster',
        'DAC',
    )

    # NOTE: point this at whatever key your config uses for CO2 units
    # (e.g. add a 'co2' entry to config_plotting["plotting"]["balance_map"]
    # analogous to the 'methanol' one, with its own unit_conversion/unit)
    conversion = config_plotting["plotting"]["balance_map"]['co2 stored']["unit_conversion"]
    bus_sizes = dac_links.groupby("location")["co2_captured"].sum().div(conversion)

    geo_scale = 0.40

    def size_to_radius(size_val):
        return np.sqrt(abs(size_val) * 2)

    # --------------------------------------------------
    # AVERAGE ONSHORE WIND CAPACITY FACTOR (per node)
    # --------------------------------------------------
    avg_onwind_cf = compute_avg_capacity_factor(n, "onwind")
    regions = regions.copy()
    # NOTE: no fillna here - nodes with no onwind generator stay NaN
    # so geopandas skips them instead of drawing them as 0.
    regions["onwind_cf"] = avg_onwind_cf.reindex(regions.index)

    vmin = regions.onwind_cf.min()  # NaNs ignored automatically
    vmax = regions.onwind_cf.max()
    if vmin == vmax:
        vmax += 1

    blue_cmap = mcolors.LinearSegmentedColormap.from_list(
        "blue_scale",
        ["#e5f0ff", "#0d47a1", "#00204d"],
    )

    crs = load_projection(copy.deepcopy(config_plotting["plotting"]))

    fig, ax = plt.subplots(
        figsize=(5, 6.5),
        subplot_kw={"projection": crs},
        layout="constrained",
    )
    ax.set_extent(boundaries, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
    regions.to_crs(crs.proj4_init).plot(
        ax=ax,
        column="onwind_cf",
        cmap=blue_cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolor="None",
        linewidth=0,
        zorder=0.5,
        missing_kwds={
            "color": "none",
        },
    )
    ax.add_feature(cfeature.COASTLINE, edgecolor="darkgrey", linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.BORDERS, edgecolor="darkgrey", linewidth=0.3, zorder=1)
    ax.spines['geo'].set_visible(False)

    grouped = dac_links.groupby(["location", "tech"])["co2_captured"].sum().reset_index().groupby("location")
    loc_coords = dac_links.groupby("location")[["x", "y"]].first()

    area_correction = get_projected_area_factor(ax, boundaries, srid=4326)

    for loc, group in grouped:
        x = float(loc_coords.loc[loc, "x"])
        y = float(loc_coords.loc[loc, "y"])
        total = group["co2_captured"].sum()
        shares = group["co2_captured"] / total
        loc_size = float(bus_sizes.loc[loc]) if loc in bus_sizes.index else 0
        if loc_size <= 0.001:
            continue

        r_corrected = size_to_radius(loc_size) * geo_scale * area_correction
        x_proj, y_proj = ax.projection.transform_point(x, y, ccrs.PlateCarree())

        tech_colors = [label_to_colors.get(tech, "#cccccc") for tech in group["tech"]]

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

    ax.set_title("DAC Captured CO₂", fontsize=8)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=blue_cmap, norm=norm)
    cbar = fig.colorbar(
        sm,
        ax=ax,
        label="Average onshore wind capacity factor [-]",
        shrink=0.95,
        pad=0.0,
        aspect=50,
        orientation="horizontal",
    )
    from matplotlib.ticker import FormatStrFormatter

    cbar.outline.set_edgecolor("None")
    cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar.set_label("Average onshore wind capacity factor [-]", fontsize=6)
    cbar.ax.tick_params(labelsize=6)

    legend_kwargs = {
        "loc": "upper left",
        "frameon": False,
        "alignment": "left",
        "fontsize": 6,
        "title_fontproperties": {
            "weight": "bold",
            "size": 7,
        },
        "handlelength": 1.0,
        "handleheight": 1.0,
        "handletextpad": 0.4,
        "labelspacing": 0.25,
        "borderaxespad": 0.2,
    }

    size_legend_kwargs = {
        "loc": "upper left",
        "frameon": False,
        "alignment": "left",
        "fontsize": 6,
        "title_fontproperties": {
            "weight": "bold",
            "size": 7,
        },
        "labelspacing": 0.4,
    }

    all_techs = ['DAC', 'DAC renewable cluster']
    tech_colors_list = [label_to_colors.get(t, "#cccccc") for t in all_techs]
    add_legend_patches(
        ax,
        tech_colors_list,
        all_techs,
        legend_kw={
            "bbox_to_anchor": (0, -0.10),
            "ncol": 1,
            "title": "DAC Captured CO2",
            **legend_kwargs,
        },
    )

    # NOTE: pick sizes appropriate to your CO2 unit scale (e.g. MtCO2)
    legend_sizes = [0.1, 1]
    carrier_unit = config_plotting["plotting"]["balance_map"]['co2 stored']["unit"]
    add_legend_semicircles(
        ax,
        [s * geo_scale**2 for s in legend_sizes],
        [f"{s} {carrier_unit}" for s in legend_sizes],
        patch_kw={"color": "#666"},
        legend_kw={
            "bbox_to_anchor": (0, 1),
            **size_legend_kwargs,
        },
    )
 

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
    # print(f'conversion: {conversion}')
    # print(f'bus_sizes {bus_sizes}')

    geo_scale = 0.15

    def size_to_radius(size_twh):
        return np.sqrt(abs(size_twh) * 2)

    crs = load_projection(copy.deepcopy(config_plotting["plotting"]))
    # print(crs)

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
    # print(f"CO2 price: {co2_price} €/tCO2")

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


def plot_load_duration_curve(n,tech_dictionary,nodes,
                             title="Cluster {node}",
                             small_capacity=70,
                             min_generator_capacity=10):
    # `title` may contain a "{node}" placeholder, which is replaced by the
    # node name of each plot (e.g. title="Renewable cluster at {node}").
    # Technologies with a capacity lower than or equal to `small_capacity`
    # [MW] are drawn in a separate bottom panel (set None for one panel).
    # Generators below `min_generator_capacity` [MW] are not plotted.

    h_per_snapshot = n.snapshot_weightings.generators.iloc[0]

    nodes = [nodes] if isinstance(nodes, str) else nodes

    for node in nodes:

        # print(f"Processing node: {node}")  # Debug statement to track which node is being processed

        load_durations=pd.DataFrame()
        pnoms={}

        if not any((node + " ")in techname for _, techname in tech_dictionary):
            continue

        for techtype, techname in tech_dictionary:

            timeserie=None
            pnom=0

            if techtype=="generator"  and ((node + " ")in techname):
                pnom=n.generators.loc[techname, "p_nom_opt"]
                if pnom < min_generator_capacity:
                    continue  # skip small generators entirely
                timeserie=(n.generators_t["p"].loc[:, techname])
                # print(f"{techname} p_nom: {pnom} MW")


            elif techtype=="link" and ((node + " ") in techname and "methanolisation" not in techname):
                timeserie=(n.links_t["p0"].loc[:, techname]).abs()
                pnom = (n.links.loc[techname,"p_nom_opt"])
                # print(f"{techname} p_nom: {pnom} MW")

            elif techtype=="link" and ((node + " ") in techname and "methanolisation" in techname):
                timeserie=(n.links_t["p2"].loc[:, techname]).abs()
                pnom = abs((n.links.loc[techname,"p_nom_opt"])*((n.links.loc[techname,"efficiency2"])))
                # print(f"{techname} p_nom: {pnom} MW")







            load_duration=timeserie

            if load_duration is None:
                continue

            techname = techname[len(node):].lstrip()
            if techtype == "generator":
                techname = techname[2:]
            techname = cluster_link_display_names.get(techname, techname)

            load_duration=load_duration.sort_values(ascending=False).reset_index(drop=True)



            if not load_duration.le(0.0001).all():
                load_durations[techname] = load_duration
                pnoms[techname] = pnom



        load_durations.index=load_durations.index*(h_per_snapshot)# convert to hours

        # print(load_durations.dtypes)
        # print(load_durations.head())


        if load_durations.empty:
            continue

        x_max = len(n.snapshots) * h_per_snapshot

        # split by size: technologies whose capacity is lower than or equal
        # to `small_capacity` [MW] go to a second panel with its own MW axis,
        # so small units (DAC, methanolisation, ...) are not flattened by the
        # large ones
        if small_capacity is None:
            large, small = list(load_durations.columns), []
        else:
            large = [c for c in load_durations.columns if pnoms[c] > small_capacity]
            # technologies below 1 MW are not plotted at all
            small = [c for c in load_durations.columns
                     if 1 <= pnoms[c] <= small_capacity]
        if large and small:
            panels = [(large, f"Large capacities (> {small_capacity:,.0f} MW)"),
                      (small, f"Small capacities (≤ {small_capacity:,.0f} MW)")]
        else:
            # only one panel left: it takes the full figure, no panel title
            panels = [(cols, "") for cols in (large, small) if cols]
        if not panels:
            continue

        fig, axes = plt.subplots(len(panels), 1, sharex=True,
                                 figsize=(12, 5.6),
                                 layout="constrained", squeeze=False)
        axes = axes[:, 0]

        nominal_power_line = Line2D([0], [0], color='#555555',
                                    linestyle=(0, (5, 3)), linewidth=1.2)

        for ax, (cols, panel_label) in zip(axes, panels):
            for techname in cols:
                color = label_to_colors.get(techname, "#cccccc")  # fallback color

                # load duration curve
                ax.plot(load_durations.index, load_durations[techname],
                        color=color, linewidth=2, label=techname, zorder=3,
                        solid_capstyle="round")

                # nominal installed capacity, same colour, dashed and lighter
                ax.hlines(y=pnoms[techname], xmin=0, xmax=x_max,
                          colors=color, linestyles=(0, (5, 3)),
                          linewidth=1.2, alpha=0.8, zorder=2)

            # axes
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, max(pnoms[c] for c in cols) * 1.1)
            ax.set_ylabel("Power [MW]")
            ax.grid(False)
            ax.set_title(panel_label, loc="left", fontsize=12)

        axes[-1].set_xlabel("hours")

        # single unified legend for all panels: the technology curves,
        # followed by one entry explaining the dashed nominal-power lines
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles += h
            labels += l
        spacer = Line2D([0], [0], color='none')
        handles += [spacer, nominal_power_line]
        labels += ['', 'Nominal installed capacity']

        fig.legend(
            handles,
            labels,
            loc='outside right center',
            frameon=False,
            fontsize=11,
            handlelength=2.5,
        )

        # title: "{node}" in the string is replaced by the node name
        if title:
            fig.suptitle(title.replace("{node}", node), x=0.0, ha="left",
                         fontsize=14, fontweight="bold")

        plt.show()

    return load_durations, pnoms


def plot_cluster_capacities(n, tech_dictionary, nodes):

    nodes = [nodes] if isinstance(nodes, str) else nodes

    for node in nodes:

        pnoms = {}

        if not any((node + " ") in techname for _, techname in tech_dictionary):
            continue

        for techtype, techname in tech_dictionary:

            pnom = None

            if techtype == "generator" and ((node + " ") in techname):
                pnom = n.generators.loc[techname, "p_nom_opt"]

            elif techtype == "link" and ((node + " ") in techname and "methanolisation" not in techname):
                pnom = n.links.loc[techname, "p_nom_opt"]

            elif techtype == "link" and ((node + " ") in techname and "methanolisation" in techname):
                pnom = abs((n.links.loc[techname, "p_nom_opt"]) * (n.links.loc[techname, "efficiency2"]))

            if pnom is None:
                continue

            techname = techname[len(node):].lstrip()
            if techtype == "generator":
                techname = techname[2:]
            techname = cluster_link_display_names.get(techname, techname)

            if pnom > 1:
                pnoms[techname] = pnom

        if not pnoms:
            continue

        capacities = pd.Series(pnoms)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [label_to_colors.get(label, "#cccccc") for label in capacities.index]
        x = range(len(capacities))
        for i, (techname, value) in enumerate(capacities.items()):
            ax.bar(i, value, color=colors[i], label=techname)

        ax.set_ylabel("Power [MW]")
        ax.set_title(f"Installed Capacities in {node} Industrial Cluster")
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.set_ylim(bottom=0)

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            frameon=True,
        )

        fig.subplots_adjust(right=0.75)

        plt.show()

    return pnoms





def plot_cluster_monthly_curtailment_share(n, node):


    generators = n.generators.index[
        n.generators.index.str.contains(node)
        & n.generators.index.str.contains("renewable cluster")
    ]
    generators = generators[n.generators.loc[generators, "p_nom_opt"].gt(1)]

    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    months = list(range(1, 13))

    if generators.empty:
        return pd.Series(0.0, index=pd.Index(months, name="month"), name=node)

    weighting = n.snapshot_weightings.generators
    snapshot_month = n.snapshots.month

    expected_production = (
        n.generators.loc[generators, "p_nom_opt"]
        * n.get_switchable_as_dense("Generator", "p_max_pu")[generators]
    ).mul(weighting, axis=0)
    actual_production = n.generators_t.p[generators].mul(weighting, axis=0)

    # total over all generators of the node
    potential_by_month = (
        expected_production.groupby(snapshot_month).sum()
        .reindex(months, fill_value=0.0)
        .sum(axis=1)
    )
    actual_by_month = (
        actual_production.groupby(snapshot_month).sum()
        .reindex(months, fill_value=0.0)
        .sum(axis=1)
    )

    produced_share = actual_by_month.div(potential_by_month).fillna(0.0)
    curtailed_share = (
        (1.0 - produced_share).clip(0.0, 1.0)
    )
    curtailed_share.index.name = "month"
    curtailed_share.name = node

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(months))
    ax.bar(
        x, produced_share * 100.0,
        color="darkblue", width=0.8, linewidth=0,
        label="Produced",
    )
    ax.bar(
        x, curtailed_share * 100.0,
        bottom=produced_share * 100.0,
        color="grey", alpha=0.5, width=0.8, linewidth=0,
        label="Curtailment",
    )
    ax.set_ylabel("Share of potential production [%]")
    ax.set_xlabel("month")
    ax.set_title(f"{node} renewable cluster monthly curtailment share")
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels)
    ax.set_ylim(0, 100)
    ax.margins(x=0.01)
    ax.legend(loc="upper right", frameon=True)
    fig.subplots_adjust(bottom=0.2)
    plt.show()

    return curtailed_share




def plot_electricity_balance_by_cluster(n, nodes, start_date=None, end_date=None, n_base=None):
    """Plot the electricity balance of the renewable cluster AC bus per node.

    For every node in ``nodes`` the generators and links whose name contains
    ``"renewable cluster"`` and the node label are collected and turned into an
    energy balance (production positive, consumption negative). Time series are
    weighted by the snapshot weightings (``n.snapshot_weightings.generators``)
    so that non-uniform snapshot spacing is handled correctly, then scaled to
    GWh. Colours are taken from ``label_to_colors``.
    """

    # hours represented by each snapshot; a Series so that variable snapshot
    # weightings are respected when converting power to energy
    h_per_snapshot = n.snapshot_weightings.generators

    el_balance_cluster = {}
    el_net = {}


    for node in nodes:

        el_balance_cluster[node]= pd.DataFrame()
        el_net[node]= pd.DataFrame()

        for generators in n.generators.loc[
            n.generators.index.str.contains('renewable cluster') & n.generators.index.str.contains(node + ' ')
        ].index:

            # print(f"Processing generator: {generators} for node: {node}")  # Debug statement


            carrier = n.generators.at[generators, "carrier"]
            clean_name = f"{carrier} renewable cluster"

            ts = (
                n.generators_t["p"].loc[:, generators] / 10**3
            )

            if clean_name in el_balance_cluster[node]:
                el_balance_cluster[node][clean_name] += ts
            else:
                el_balance_cluster[node][clean_name] = ts


        for links in n.links.loc[n.links.index.str.contains('renewable cluster') & n.links.index.str.contains(node + ' ')].index:

            if node in links:
                clean_name = links[len(node):].lstrip()
                clean_name = cluster_link_display_names.get(clean_name, clean_name)




                if "methanolisation" in links:
                    # methanolisation electricity input is on bus2
                    el_balance_cluster[node][clean_name]= (n.links_t["p2"].loc[:, links]/-10**3)
                    el_balance_cluster[node][clean_name] = el_balance_cluster[node][clean_name].fillna(0)
                elif "Electrolysis" in links:
                    el_balance_cluster[node][clean_name]= (n.links_t["p0"].loc[:, links]/-10**3)
                    el_balance_cluster[node][clean_name] = el_balance_cluster[node][clean_name].fillna(0)
                elif "DAC" in links:
                    # DAC electricity input is on bus0
                    el_balance_cluster[node][clean_name]= (n.links_t["p0"].loc[:, links]/-10**3)
                    el_balance_cluster[node][clean_name] = el_balance_cluster[node][clean_name].fillna(0)
                elif "Fischer-Tropsch" in links:
                    # Fischer-Tropsch has no electricity bus (bus0=H2, bus1=oil,
                    # bus2=co2 stored), so it does not enter the electricity balance
                    continue
                elif "battery charger" in links:
                    # charger electricity input is on bus0 (p1 is already reduced
                    # by the inverter efficiency, so it would not balance)
                    el_balance_cluster[node][clean_name]= (n.links_t["p0"].loc[:, links]/-10**3)
                    el_balance_cluster[node][clean_name] = el_balance_cluster[node][clean_name].fillna(0)
                elif "battery discharger" in links:
                    el_balance_cluster[node][clean_name]= (n.links_t["p1"].loc[:, links]/-10**3)
                    el_balance_cluster[node][clean_name] = el_balance_cluster[node][clean_name].fillna(0)
                elif f"{node} electricity renewable cluster"==links:
                    el_balance_cluster[node][clean_name]= (n.links_t["p0"].loc[:, links]/-10**3)
                    el_balance_cluster[node][clean_name] = el_balance_cluster[node][clean_name].fillna(0)
                elif f"{node} electricity renewable cluster back"==links:
                    el_balance_cluster[node][clean_name]= (n.links_t["p0"].loc[:, links]/10**3)
                    el_balance_cluster[node][clean_name] = el_balance_cluster[node][clean_name].fillna(0)
                elif f"{node} electricity renewable cluster both"==links:
                    # bidirectional grid link carries both signs, which breaks the
                    # stacked area plot; split it into two single-signed columns.
                    # positive p0 -> grid connection (buy),
                    # negative p0 -> grid connection (sell)
                    # (names must match the keys in label_to_colors)
                    ts = (n.links_t["p0"].loc[:, links]/10**3)
                    el_balance_cluster[node]["grid connection (buy) renewable cluster"] = ts.clip(lower=0).fillna(0)
                    el_balance_cluster[node]["grid connection (sell) renewable cluster"] = ts.clip(upper=0).fillna(0)

        # print(el_balance_cluster[node])
        # net balance from the aggregated series, computed before masking so it
        # reflects the true (near-zero) bus balance
        el_net[node]["el_net"] = el_balance_cluster[node].sum(axis=1)
        el_net[node].index = el_balance_cluster[node].index

        negligible_cols = el_balance_cluster[node].columns[
            el_balance_cluster[node].abs().max() <= 0.001
        ]
        el_balance_cluster[node] = el_balance_cluster[node].drop(columns=negligible_cols)

        # # hide negligible values for a cleaner area plot
        # threshold = 1e-4  # adjust if needed
        # el_balance_cluster[node] = el_balance_cluster[node].where(
        #     el_balance_cluster[node].abs() > threshold
        # )


        col_order = el_balance_cluster[node].abs().max().sort_values().index
        el_balance_cluster[node] = el_balance_cluster[node][col_order]

        # only plot the requested window, and drop technologies that are
        # negligible within it so the legend only lists what is visible
        plot_df = el_balance_cluster[node].loc[start_date:end_date]
        plot_df = plot_df.loc[:, plot_df.abs().max() > 0.001]
        net = el_net[node]["el_net"].loc[start_date:end_date]

        figure, ax = plt.subplots(figsize=(13, 6.5))

        plot_df.plot.area(
            ax=ax,
            linewidth=0,
            alpha=0.9,
            # use plain matplotlib dates (not pandas' period axis) so the areas
            # share the x-axis with ax.plot / set_xlim / mdates formatting
            x_compat=True,
            color=[label_to_colors.get(label, "#cccccc") for label in plot_df.columns],
            legend=False,
        )
        ax.plot(net.index, net,
                color=label_to_colors['Net Balance'],
                linewidth=1.5,
                zorder=10,
                label='Net balance')
        ax.axhline(0, color="black", linewidth=0.8, zorder=5)

        # electricity prices on a second y-axis on the right; only the plotted
        # window is drawn so the price axis scales to what is shown
        ax_price = ax.twinx()
        el_price = n.buses_t.marginal_price.loc[:, node].loc[start_date:end_date]
        ax_price.plot(el_price.index, el_price,
                      color='red',
                      linewidth=1.5,
                      label='Electricity price outside the cluster')
        # same price from the baseline network (no industrial clusters)
        if n_base is not None:
            el_price_base = n_base.buses_t.marginal_price.loc[:, node].loc[start_date:end_date]
            ax_price.plot(el_price_base.index, el_price_base,
                          '--',
                          color='red',
                          alpha=0.6,
                          linewidth=1.5,
                          label='Electricity price outside the cluster (baseline)')
        # electricity price at the cluster's own AC bus
        el_price_cluster = n.buses_t.marginal_price.loc[:, f"{node} renewable cluster"].loc[start_date:end_date]
        ax_price.plot(el_price_cluster.index, el_price_cluster,
                      color='navy',
                      linewidth=1.5,
                      label='Electricity price inside the cluster')
        ax_price.set_ylabel("Electricity price [€/MWh]")
        ax_price.set_ylim(bottom=min(0, ax_price.get_ylim()[0]))

        # Layout / labels
        ax.set_xlim(plot_df.index[0], plot_df.index[-1])
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.set_ylabel("Power [GW]\n← consumption | production →")
        ax.set_title(f"Electricity balance of the renewable cluster at {node}",
                     fontsize=13, fontweight="bold")
        for spine in ("top",):
            ax.spines[spine].set_visible(False)
            ax_price.spines[spine].set_visible(False)

        # one combined legend below the plot; drop the long
        # " renewable cluster" suffix from the technology names
        handles, labels = ax.get_legend_handles_labels()
        price_handles, price_labels = ax_price.get_legend_handles_labels()
        labels = [l.replace(" renewable cluster", "") for l in labels]
        ax.legend(handles + price_handles, labels + price_labels,
                  loc="upper center", bbox_to_anchor=(0.5, -0.08),
                  ncol=3, frameon=False, fontsize=9)
        figure.tight_layout()

    return el_balance_cluster


def plot_cluster_self_supply_violin(n, node):
    """Violin plot of the hourly share of the cluster electricity supply that
    comes from inside the cluster.

    The balance is recomputed with the same sign conventions as
    ``plot_electricity_balance_by_cluster``:
      - cluster electricity production = everything on the positive side
      - internal production            = positive side minus grid connection (buy)
    Each snapshot is one dot. The black bar marks the time-weighted average
    share, i.e. sum(share * w) / sum(w) with w the snapshot weightings, over the
    snapshots with non-zero supply.
    """

    # only meaningful if the cluster actually runs an electrolyser:
    # yearly electrolyser electricity consumption must exceed 5 GWh
    electrolysers = n.links.index[
        n.links.index.str.contains("renewable cluster")
        & n.links.index.str.contains(node + " ")
        & n.links.index.str.contains("Electrolysis")
    ]
    electrolyser_consumption = (
        n.links_t["p0"][electrolysers].clip(lower=0)
        .mul(n.snapshot_weightings.generators, axis=0)
        .sum().sum() / 10**3
    )  # GWh
    if electrolyser_consumption <= 5:
        print(f"the cluster in {node} is a hybrid powerplant. The violin plot doesn't make sense")
        return None, None

    balance = {}  # column -> time series [GW], production positive
    grid_buy = pd.Series(0.0, index=n.snapshots)

    gens = n.generators.index[
        n.generators.index.str.contains("renewable cluster")
        & n.generators.index.str.contains(node + " ")
    ]
    for gen in gens:
        balance[gen] = n.generators_t["p"][gen] / 10**3

    links = n.links.index[
        n.links.index.str.contains("renewable cluster")
        & n.links.index.str.contains(node + " ")
    ]
    for link in links:
        if "methanolisation" in link:
            ts = n.links_t["p2"][link] / -10**3
        elif "Fischer-Tropsch" in link:
            continue  # no electricity bus
        elif "battery discharger" in link:
            ts = n.links_t["p1"][link] / -10**3
        elif ("Electrolysis" in link or "DAC" in link or "battery charger" in link
              or link == f"{node} electricity renewable cluster"):
            ts = n.links_t["p0"][link] / -10**3
        elif link == f"{node} electricity renewable cluster back":
            ts = n.links_t["p0"][link] / 10**3
        elif link == f"{node} electricity renewable cluster both":
            # positive p0 = grid connection (buy), negative = (sell)
            ts = n.links_t["p0"][link] / 10**3
            grid_buy = grid_buy + ts.clip(lower=0).fillna(0)
            balance[link + " (sell)"] = ts.clip(upper=0)
            continue
        else:
            continue
        balance[link] = ts

    balance = pd.DataFrame(balance, index=n.snapshots).fillna(0)

    positive = balance.clip(lower=0)
    internal = positive.sum(axis=1)
    total = internal + grid_buy

    # snapshots with no supply have an undefined share
    valid = total > 1e-6
    share = (internal[valid] / total[valid] * 100.0).clip(0.0, 100.0)

    # time-weighted mean of the share over the snapshots with supply:
    # sum(share * w) / sum(w), w = hours represented by each snapshot
    w = n.snapshot_weightings.generators.loc[share.index]
    weighted_avg = (share * w).sum() / w.sum()

    fig, ax = plt.subplots(figsize=(5, 6))
    parts = ax.violinplot(share.values, positions=[0], widths=0.8,
                          showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#C8A2C8")
        body.set_edgecolor("none")
        body.set_alpha(0.6)

    # one dot per snapshot, jittered horizontally inside the violin
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.15, 0.15, size=len(share))
    ax.scatter(jitter, share.values, s=4, color="#9B6FB0", alpha=0.5,
               linewidths=0, zorder=3)

    # weighted average as a short black bar
    ax.hlines(weighted_avg, -0.12, 0.12, color="black", linewidth=3, zorder=5,
              label=f"Time-weighted average: {weighted_avg:.1f} %")

    ax.set_xticks([0])
    ax.set_xticklabels([node])
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Share of cluster electricity supplied\nfrom inside the cluster [%]")
    ax.set_title(f"Hourly self-supply share of the renewable cluster at {node}",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    fig.tight_layout()
    plt.show()

    return share, weighted_avg


def plot_cluster_price_duration_curves(n, node, max_electricity_price, n_base=None, save_path=None,
                                       max_hours=8760):
    """Price duration curves of the electricity prices shown in
    ``plot_electricity_balance_by_cluster``:
      - inside the cluster   (bus "<node> renewable cluster" in ``n``)
      - outside the cluster  (bus "<node>" in ``n``)
      - outside the cluster, baseline (bus "<node>" in ``n_base``, if given)
    Prices are sorted from highest to lowest; the x axis is the cumulative
    number of hours (snapshot weightings), so it runs from 0 to 8760.
    Prices above ``max_electricity_price`` are plotted as ``max_electricity_price``.
    If ``save_path`` is given, the figure is saved there (e.g. as a PDF).
    ``max_hours`` limits the x axis, e.g. max_hours=1000 zooms on the
    1000 hours with the highest prices.
    """

    # only meaningful if the cluster actually runs an electrolyser:
    # yearly electrolyser electricity consumption must exceed 5 GWh
    electrolysers = n.links.index[
        n.links.index.str.contains("renewable cluster")
        & n.links.index.str.contains(node + " ")
        & n.links.index.str.contains("Electrolysis")
    ]
    electrolyser_consumption = (
        n.links_t["p0"][electrolysers].clip(lower=0)
        .mul(n.snapshot_weightings.generators, axis=0)
        .sum().sum() / 10**3
    )  # GWh
    # hybrid powerplant: the price inside the cluster is not meaningful,
    # so only the prices outside the cluster are plotted
    hybrid = electrolyser_consumption <= 5
    if hybrid:
        print(f"the cluster in {node} is a hybrid powerplant")

    def duration_curve(network, bus):
        price = network.buses_t.marginal_price[bus]
        w = network.snapshot_weightings.generators.loc[price.index]
        order = price.sort_values(ascending=False).index
        sorted_price = price.loc[order].clip(upper=max_electricity_price)
        hours = np.concatenate([[0.0], w.loc[order].cumsum().values])
        # repeat the last value so each snapshot spans its full weighting
        return hours, np.append(sorted_price.values, sorted_price.values[-1])

    curves = []
    if not hybrid:
        curves.append((n, f"{node} renewable cluster", "Electricity price inside the cluster",
                       dict(color="navy", linestyle="-")))
    curves.append((n, node, "Electricity price outside the cluster",
                   dict(color="red", linestyle="-")))
    if n_base is not None:
        curves.append((n_base, node, "Electricity price outside the cluster (baseline)",
                       dict(color="red", linestyle="--", alpha=0.6)))

    fig, ax = plt.subplots(figsize=(10, 5))
    duration_curves = {}
    for network, bus, label, style in curves:
        hours, prices = duration_curve(network, bus)
        ax.step(hours, prices, where="post", linewidth=1.5, label=label, **style)
        duration_curves[label] = pd.Series(prices[:-1], index=hours[:-1])

    ax.axhline(max_electricity_price, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlim(0, max_hours)
    # y axis fitted to the prices shown in the zoomed window
    visible_min = min(s[s.index < max_hours].min() for s in duration_curves.values())
    ax.set_ylim(bottom=min(0, visible_min * 1.05), top=max_electricity_price * 1.05)
    ax.set_xlabel("Hours of the year")
    ax.set_ylabel("Electricity price [€/MWh]")
    ax.set_title(f"Electricity price duration curves at {node}",
                 fontsize=13, fontweight="bold")
    ax.grid(linestyle=":", alpha=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()

    return duration_curves


def plot_methanol_merit_order(n, min_capacity_el=1):
    """Merit order of the average methanol prices: one bar per production
    site, sorted from cheapest to most expensive. Sites inside the renewable
    cluster are labelled "<node> cluster".

    Only sites whose methanolisation capacity exceeds ``min_capacity_el`` [MWel]
    are shown. p_nom_opt is in MW of hydrogen input (bus0), so it is converted
    to MWel with the electricity input on bus2: p_nom_opt * |efficiency2|
    (same conversion as in plot_load_duration_curve).
    """

    _, _, _, methanol_avg = compute_ft_and_methanol_prices(n)
    _, _, _, methanol_avg_rc, _ = compute_ft_and_methanol_prices_renewable_cluster(n)

    prices = {}
    for avg, suffix, label_suffix in [
        (methanol_avg, "", ""),
        (methanol_avg_rc, " renewable cluster", " cluster"),
    ]:
        for node, price in avg.dropna().items():
            link = f"{node} methanolisation{suffix}"
            if link not in n.links.index:
                continue
            capacity_el = abs(n.links.at[link, "p_nom_opt"] * n.links.at[link, "efficiency2"])
            if capacity_el > min_capacity_el:
                prices[node + label_suffix] = price

    prices = pd.Series(prices, dtype=float).sort_values()

    if prices.empty:
        print(f"No methanolisation plant above {min_capacity_el} MWel in this network")
        return prices

    colors = ["navy" if label.endswith(" cluster") else "darkorange" for label in prices.index]

    # cheapest on the left, most expensive on the right
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(prices)), 5.5))
    x = np.arange(len(prices))
    bars = ax.bar(x, prices.values, color=colors, width=0.45, linewidth=0, zorder=3)
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(prices.index, rotation=60, ha="right", rotation_mode="anchor", fontsize=9)
    ax.set_xlim(-0.6, len(prices) - 0.4)
    ax.set_ylim(min(0, prices.min() * 1.1), prices.max() * 1.18)
    ax.set_xlabel("Production site (cheapest → most expensive)")
    ax.set_ylabel("Average methanol price [€/MWh]")
    ax.set_title("Merit order of methanol production prices",
                 fontsize=13, fontweight="bold", pad=22)
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(
        handles=[
            mpatches.Patch(color="darkorange", label="Outside the cluster"),
            mpatches.Patch(color="navy", label="Inside the cluster"),
        ],
        # centred above the axes, between the title and the bars
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
        frameon=False, fontsize=9,
    )
    fig.tight_layout()
    plt.show()

    return prices


def plot_FT_merit_order(n, min_capacity=1):
    """Merit order of the average Fischer-Tropsch oil prices: one bar per
    production site, sorted from cheapest to most expensive. Sites inside the
    renewable cluster are labelled "<node> cluster".

    Fischer-Tropsch has no electricity input (bus0 = H2, bus1 = oil,
    bus2 = co2 stored), so the capacity filter is on p_nom_opt in MW of
    hydrogen input: only sites above ``min_capacity`` [MW_H2] are shown.
    """

    FT_avg = compute_ft_and_methanol_prices(n)[2]
    FT_avg_rc = compute_ft_and_methanol_prices_renewable_cluster(n)[2]

    prices = {}
    for avg, suffix, label_suffix in [
        (FT_avg, "", ""),
        (FT_avg_rc, " renewable cluster", " cluster"),
    ]:
        for node, price in avg.dropna().items():
            link = f"{node} Fischer-Tropsch{suffix}"
            if link not in n.links.index:
                continue
            if n.links.at[link, "p_nom_opt"] > min_capacity:
                prices[node + label_suffix] = price

    prices = pd.Series(prices, dtype=float).sort_values()

    if prices.empty:
        print(f"No Fischer-Tropsch plant above {min_capacity} MW_H2 in this network")
        return prices

    colors = ["navy" if label.endswith(" cluster") else "darkorange" for label in prices.index]

    # cheapest on the left, most expensive on the right
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(prices)), 5.5))
    x = np.arange(len(prices))
    bars = ax.bar(x, prices.values, color=colors, width=0.45, linewidth=0, zorder=3)
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(prices.index, rotation=60, ha="right", rotation_mode="anchor", fontsize=9)
    ax.set_xlim(-0.6, len(prices) - 0.4)
    ax.set_ylim(min(0, prices.min() * 1.1), prices.max() * 1.18)
    ax.set_xlabel("Production site (cheapest → most expensive)")
    ax.set_ylabel("Average Fischer-Tropsch oil price [€/MWh]")
    ax.set_title("Merit order of Fischer-Tropsch oil production prices",
                 fontsize=13, fontweight="bold", pad=22)
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(
        handles=[
            mpatches.Patch(color="darkorange", label="Outside the cluster"),
            mpatches.Patch(color="navy", label="Inside the cluster"),
        ],
        # centred above the axes, between the title and the bars
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
        frameon=False, fontsize=9,
    )
    fig.tight_layout()
    plt.show()

    return prices


def plot_cheapest_methanol_cost_breakdown(n, min_capacity_el=1, save_path=None):
    """Stacked bars with the cost breakdown of the average methanol price of
    the cheapest methanol producer inside a renewable cluster and the
    cheapest one outside the clusters.

    Components follow the price formulas in compute_ft_and_methanol_prices
    (outside) and compute_ft_and_methanol_prices_renewable_cluster (cluster):
    each term is summed over snapshots and divided by total methanol
    production, so the components add up to the average price. The CO2
    component includes the CO2 allowance term, heat is a credit (negative).
    Only plants above ``min_capacity_el`` [MWel] (p_nom_opt * |efficiency2|)
    are considered.
    """

    methanol_avg = compute_ft_and_methanol_prices(n)[3]
    methanol_avg_rc = compute_ft_and_methanol_prices_renewable_cluster(n)[3]

    weights = n.snapshot_weightings.generators
    co2_intensity_methanol = n.links.loc[
        n.links.index.str.contains("EU shipping methanol"), 'efficiency2'
    ].iloc[0]
    co2_price = n.global_constraints.loc[
        n.global_constraints.carrier_attribute.str.contains("co2_emissions"), 'mu'
    ].iloc[0]

    breakdowns = {}
    for avg, suffix, label_suffix, group in [
        (methanol_avg_rc, " renewable cluster", " cluster", "the renewable clusters"),
        (methanol_avg, "", "", "the nodes outside the clusters"),
    ]:
        # cheapest producer above the capacity threshold
        candidates = {}
        for node, price in avg.dropna().items():
            link = f"{node} methanolisation{suffix}"
            if link in n.links.index and abs(
                n.links.at[link, "p_nom_opt"] * n.links.at[link, "efficiency2"]
            ) > min_capacity_el:
                candidates[node] = price
        if not candidates:
            print(f"None of {group} produces methanol in this network")
            continue
        node = pd.Series(candidates).idxmin()
        link = f"{node} methanolisation{suffix}"

        O_and_M_cost = n.links.at[link, 'marginal_cost']
        capital_cost = n.links.at[link, 'capital_cost'] * n.links.at[link, 'p_nom_opt']

        price_of_H2 = n.buses_t.marginal_price.loc[:, f"{node} H2{suffix}"]
        consumption_of_H2 = n.links_t.p0.loc[:, link]
        consumption_of_H2_total = consumption_of_H2.multiply(weights, axis=0).sum()
        capital_cost_allocated = capital_cost * consumption_of_H2 / consumption_of_H2_total

        price_of_CO2 = n.buses_t.marginal_price.loc[:, f"{node} co2 stored{suffix}"]
        consumption_of_CO2 = n.links_t.p3.loc[:, link]
        price_of_electricity = n.buses_t.marginal_price.loc[:, f"{node}{suffix}"]
        consumption_of_electricity = n.links_t.p2.loc[:, link]

        heat_term = 0.0
        if f"{node} urban central heat{suffix}" in n.buses_t.marginal_price.columns:
            price_of_heat = n.buses_t.marginal_price.loc[:, f"{node} urban central heat{suffix}"]
            production_of_heat = n.links_t.p4.loc[:, link].abs()
            heat_term = (price_of_heat * production_of_heat).sum()

        production_total = n.links_t.p1.loc[:, link].abs().sum()

        group_label = "Cheapest cluster" if suffix else "Cheapest outside the clusters"
        breakdowns[f"{group_label}\n{node}{label_suffix}"] = pd.Series({
            "Hydrogen": (price_of_H2 * consumption_of_H2).sum() / production_total,
            "Electricity": (price_of_electricity * consumption_of_electricity).sum() / production_total,
            "CO2": (price_of_CO2 * consumption_of_CO2).sum() / production_total
                   - co2_intensity_methanol * co2_price,
            "Capital": capital_cost_allocated.sum() / production_total,
            "O&M": (O_and_M_cost * consumption_of_H2).sum() / production_total,
            "Heat": -heat_term / production_total,
        })

    if not breakdowns:
        return None

    breakdowns = pd.DataFrame(breakdowns)  # rows: components, columns: producers

    component_colors = {
        "Hydrogen": "#4C9BE8",
        "Electricity": "#F2C14E",
        "CO2": "#7F7F7F",
        "Capital": "#8E6BBF",
        "O&M": "#5DB58A",
        "Heat": "#E07A5F",
    }

    fig, ax = plt.subplots(figsize=(6 + 1.5 * breakdowns.shape[1], 6))
    x = np.arange(breakdowns.shape[1])
    width = 0.4

    # stack positive components upwards and negative ones (credits) downwards
    pos_bottom = np.zeros(len(x))
    neg_bottom = np.zeros(len(x))
    for comp in breakdowns.index:
        values = breakdowns.loc[comp].values.astype(float)
        if np.allclose(values, 0):
            continue
        bottom = np.where(values >= 0, pos_bottom, neg_bottom)
        ax.bar(x, values, width=width, bottom=bottom,
               color=component_colors.get(comp, "#cccccc"),
               linewidth=0.5, edgecolor="white", label=comp, zorder=3)
        # value inside the segment if it is large enough to read
        for xi, v, b in zip(x, values, bottom):
            if abs(v) > 0.04 * breakdowns.clip(lower=0).sum().max():
                ax.text(xi, b + v / 2, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black", zorder=4)
        pos_bottom += np.clip(values, 0, None)
        neg_bottom += np.clip(values, None, 0)

    # total = average price
    totals = breakdowns.sum()
    ax.scatter(x, totals.values, marker="D", color="black", s=30, zorder=5,
               label="Average price")
    for xi, t in zip(x, totals.values):
        ax.text(xi + width / 2 + 0.05, t, f"{t:.2f} €/MWh",
                va="center", fontsize=9, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(breakdowns.columns)
    ax.set_xlim(-0.6, len(x) - 0.4 + 0.3)
    ax.set_ylabel("Cost contribution [€/MWh of methanol]")
    ax.set_title("Cost breakdown of the cheapest methanol producers",
                 fontsize=13, fontweight="bold", pad=40)
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
              frameon=False, fontsize=9)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()

    return breakdowns


def plot_cheapest_FT_cost_breakdown(n, min_capacity=1, save_path=None):
    """Stacked bars with the cost breakdown of the average Fischer-Tropsch
    oil price of the cheapest FT producer inside a renewable cluster and the
    cheapest one outside the clusters.

    Components follow the price formulas in compute_ft_and_methanol_prices
    (outside) and compute_ft_and_methanol_prices_renewable_cluster (cluster):
    each term is summed over snapshots and divided by total FT oil
    production, so the components add up to the average price. The CO2
    component includes the CO2 allowance term, heat is a credit (negative).
    FT has no electricity input, so only plants above ``min_capacity``
    [MW_H2] are considered.
    """

    FT_avg = compute_ft_and_methanol_prices(n)[2]
    FT_avg_rc = compute_ft_and_methanol_prices_renewable_cluster(n)[2]

    weights = n.snapshot_weightings.generators
    co2_price = n.global_constraints.loc[
        n.global_constraints.carrier_attribute.str.contains("co2_emissions"), 'mu'
    ].iloc[0]

    is_FT = n.links.index.str.contains("Fischer-Tropsch")
    is_rc = n.links.index.str.contains("renewable cluster")

    breakdowns = {}
    for avg, suffix, label_suffix, group, FT_mask in [
        (FT_avg_rc, " renewable cluster", " cluster", "the renewable clusters", is_FT & is_rc),
        # same selection as in compute_ft_and_methanol_prices
        (FT_avg, "", "", "the nodes outside the clusters", is_FT),
    ]:
        # cheapest producer above the capacity threshold
        candidates = {}
        for node, price in avg.dropna().items():
            link = f"{node} Fischer-Tropsch{suffix}"
            if link in n.links.index and n.links.at[link, "p_nom_opt"] > min_capacity:
                candidates[node] = price
        if not candidates:
            print(f"None of {group} produces Fischer-Tropsch oil in this network")
            continue
        node = pd.Series(candidates).idxmin()
        link = f"{node} Fischer-Tropsch{suffix}"

        co2_intensity_FT = (
            n.links.loc[FT_mask, 'efficiency2'].iloc[0]
            / n.links.loc[FT_mask, 'efficiency'].iloc[0]
        )

        O_and_M_cost = n.links.at[link, 'marginal_cost']
        capital_cost = n.links.at[link, 'capital_cost'] * n.links.at[link, 'p_nom_opt']

        price_of_H2 = n.buses_t.marginal_price.loc[:, f"{node} H2{suffix}"]
        consumption_of_H2 = n.links_t.p0.loc[:, link]
        consumption_of_H2_total = consumption_of_H2.multiply(weights, axis=0).sum()
        capital_cost_allocated = capital_cost * consumption_of_H2 / consumption_of_H2_total

        price_of_CO2 = n.buses_t.marginal_price.loc[:, f"{node} co2 stored{suffix}"]
        consumption_of_CO2 = n.links_t.p2.loc[:, link]

        heat_term = 0.0
        if f"{node} urban central heat{suffix}" in n.buses_t.marginal_price.columns:
            price_of_heat = n.buses_t.marginal_price.loc[:, f"{node} urban central heat{suffix}"]
            production_of_heat = n.links_t.p3.loc[:, link].abs()
            heat_term = (price_of_heat * production_of_heat).sum()

        production_total = n.links_t.p1.loc[:, link].abs().sum()

        group_label = "Cheapest cluster" if suffix else "Cheapest outside the clusters"
        breakdowns[f"{group_label}\n{node}{label_suffix}"] = pd.Series({
            "Hydrogen": (price_of_H2 * consumption_of_H2).sum() / production_total,
            "CO2": (price_of_CO2 * consumption_of_CO2).sum() / production_total
                   + co2_intensity_FT * co2_price,
            "Capital": capital_cost_allocated.sum() / production_total,
            "O&M": (O_and_M_cost * consumption_of_H2).sum() / production_total,
            "Heat": -heat_term / production_total,
        })

    if not breakdowns:
        return None

    breakdowns = pd.DataFrame(breakdowns)  # rows: components, columns: producers

    component_colors = {
        "Hydrogen": "#4C9BE8",
        "CO2": "#7F7F7F",
        "Capital": "#8E6BBF",
        "O&M": "#5DB58A",
        "Heat": "#E07A5F",
    }

    fig, ax = plt.subplots(figsize=(6 + 1.5 * breakdowns.shape[1], 6))
    x = np.arange(breakdowns.shape[1])
    width = 0.4

    # stack positive components upwards and negative ones (credits) downwards
    pos_bottom = np.zeros(len(x))
    neg_bottom = np.zeros(len(x))
    for comp in breakdowns.index:
        values = breakdowns.loc[comp].values.astype(float)
        if np.allclose(values, 0):
            continue
        bottom = np.where(values >= 0, pos_bottom, neg_bottom)
        ax.bar(x, values, width=width, bottom=bottom,
               color=component_colors.get(comp, "#cccccc"),
               linewidth=0.5, edgecolor="white", label=comp, zorder=3)
        # value inside the segment if it is large enough to read
        for xi, v, b in zip(x, values, bottom):
            if abs(v) > 0.04 * breakdowns.clip(lower=0).sum().max():
                ax.text(xi, b + v / 2, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black", zorder=4)
        pos_bottom += np.clip(values, 0, None)
        neg_bottom += np.clip(values, None, 0)

    # total = average price
    totals = breakdowns.sum()
    ax.scatter(x, totals.values, marker="D", color="black", s=30, zorder=5,
               label="Average price")
    for xi, t in zip(x, totals.values):
        ax.text(xi + width / 2 + 0.05, t, f"{t:.2f} €/MWh",
                va="center", fontsize=9, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(breakdowns.columns)
    ax.set_xlim(-0.6, len(x) - 0.4 + 0.3)
    ax.set_ylabel("Cost contribution [€/MWh of FT oil]")
    ax.set_title("Cost breakdown of the cheapest Fischer-Tropsch producers",
                 fontsize=13, fontweight="bold", pad=40)
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
              frameon=False, fontsize=9)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()

    return breakdowns


def plot_cluster_electricity_balance_map(
    n,
    regions,
    config_plotting,
    label_to_colors,
    boundaries,
    nodes=None,
    unit="TWh",
    bus_size_factor=0.01,
    legend_bus_sizes=(10, 20),
    min_branch_width=0.3,
    branch_width_scale=2,
    loading_threshold=0.98,
):
    """AC balance map (as plot_balance_map.py) with the cluster balance.

    Lines, links and flows are drawn exactly as in the pypsa-eur AC balance
    map, without the regional price colormap; only the region borders of
    `regions` are drawn. The semicircles show the annual electricity
    balance of the '<node> renewable cluster' bus (same bookkeeping as
    plot_electricity_balance_by_cluster), drawn at the node of the cluster.
    """

    SEMICIRCLE_CORRECTION_FACTOR = (
        2 if parse(pypsa.__version__) <= Version("0.33.2")
        else 1
    )
    settings = config_plotting["plotting"]["balance_map"]["AC"]
    weights = n.snapshot_weightings.generators

    # AC buses outside the cluster
    ac_buses = n.buses.index[
        (n.buses.carrier == "AC") & ~n.buses.index.str.endswith(" renewable cluster")
    ]
    if nodes is None:
        nodes = [b for b in ac_buses if f"{b} renewable cluster" in n.buses.index]

    # --------------------------------------------------
    # Annual electricity balance of each cluster bus [MWh]
    # --------------------------------------------------
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
    bus_size = bus_size.sort_values(ascending=False)

    # --------------------------------------------------
    # Colours (legend labels without the ' renewable cluster' suffix)
    # --------------------------------------------------
    carrier_colors = n.carriers.color.copy().replace("", "grey")

    def get_color(c):
        return label_to_colors.get(c, carrier_colors.get(c, "lightgrey"))

    def get_label(c):
        return c.replace(" renewable cluster", "")

    colors = (
        bus_size.index.get_level_values("carrier")
        .unique()
        .to_series()
        .map(get_color)
    )

    # --------------------------------------------------
    # AC line / link flows (as in plot_balance_map.py)
    # --------------------------------------------------
    branch_conversion = settings["unit_conversion"]
    flow = n.statistics.transmission(groupby=False, bus_carrier="AC").div(branch_conversion)
    if not flow.empty:
        flow_reversed_mask = flow.index.get_level_values(1).str.contains("reversed")
        flow_reversed = flow[flow_reversed_mask].rename(
            lambda x: x.replace("-reversed", "")
        )
        flow = flow[~flow_reversed_mask].subtract(flow_reversed, fill_value=0)

    fallback = pd.Series(dtype=float)
    line_widths = flow.get("Line", fallback).abs()
    link_widths = flow.get("Link", fallback).abs()
    line_flow = flow.get("Line")
    link_flow = flow.get("Link")
    transformer_flow = flow.get("Transformer")
    # scale the config width so the map lines and the legend stay consistent
    branch_width_factor = settings["branch_factor"] * branch_width_scale
    flow_size_factor = settings["flow_factor"]

    # --------------------------------------------------
    # Line colours: hours per year with loading >= loading_threshold
    # --------------------------------------------------
    # AC lines: loading = p0 if > 0 else p1, over s_max_pu * s_nom_opt
    cap = n.lines.s_nom_opt * n.lines.s_max_pu
    p0 = n.lines_t.p0
    ac_loading = p0.where(p0 > 0, n.lines_t.p1).div(cap, axis=1)

    # DC lines: forward link '<name>' plus '<name>-reversed' link;
    # loading = forward p0 if >= 1 else reversed p0, over forward p_nom_opt
    dc_fwd = n.links.index[
        (n.links.carrier == "DC") & ~n.links.index.str.endswith("-reversed")
    ]
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
    hours_98 = (
        loading.ge(loading_threshold)
        .mul(n.snapshot_weightings.objective, axis=0)
        .sum()
    )

    line_cmap = mcolors.LinearSegmentedColormap.from_list(
        "red_scale",
        ["#ffe5e5", "#b30000", "#4d0000"],
    )
    line_norm = plt.Normalize(vmin=0, vmax=max(hours_98.max(), 1))

    def hours_to_color(h):
        return mcolors.to_hex(line_cmap(line_norm(h)))

    branch_colors = hours_98.map(hours_to_color)
    line_colors = branch_colors["Line"].reindex(n.lines.index, fill_value=hours_to_color(0))

    # links on the map are indexed by the forward name (reversed merged in);
    # non-DC links keep the default pypsa link colour
    link_colors = pd.Series("darkseagreen", index=link_widths.index, dtype=object)
    dc_on_map = dc_fwd.intersection(link_colors.index)
    link_colors[dc_on_map] = branch_colors["Link"][dc_on_map]

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------
    crs = load_projection(copy.deepcopy(config_plotting["plotting"]))
    fig, ax = plt.subplots(
        figsize=(5, 6.5),
        subplot_kw={"projection": crs},
        layout="constrained",
    )

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
    dc_dash = (0, (1, 1.5))
    link_coll = collections["branches"].get("Link")
    if link_coll is not None and len(link_coll.get_segments()) == len(link_widths):
        link_coll.set_linestyle(
            [dc_dash if l in dc_on_map else "solid" for l in link_widths.index]
        )
        link_coll.set_capstyle("round")

    ax.set_title("AC (renewable cluster balance)")

    sm = plt.cm.ScalarMappable(cmap=line_cmap, norm=line_norm)
    cbr = fig.colorbar(
        sm,
        ax=ax,
        label=f"Hours with line loading ≥ {loading_threshold:.0%} [h]",
        shrink=0.95,
        pad=0.03,
        aspect=50,
        orientation="horizontal",
    )
    cbr.outline.set_edgecolor("None")

    # --------------------------------------------------
    # Legends
    # --------------------------------------------------
    legend_kwargs = {
        "loc": "upper left",
        "frameon": False,
        "alignment": "left",
        "title_fontproperties": {"weight": "bold"},
    }
    pad = 0.18

    pos_carriers = bus_size[bus_size > 0].index.unique("carrier")
    neg_carriers = bus_size[bus_size < 0].index.unique("carrier")
    common_carriers = pos_carriers.intersection(neg_carriers)

    def get_total_abs(c, sign):
        vals = bus_size.xs(c, level="carrier")
        return vals[vals * sign > 0].abs().sum()

    supp_carriers = sorted(
        set(pos_carriers) - set(common_carriers)
        | {c for c in common_carriers if get_total_abs(c, 1) >= get_total_abs(c, -1)}
    )
    cons_carriers = sorted(
        set(neg_carriers) - set(common_carriers)
        | {c for c in common_carriers if get_total_abs(c, 1) < get_total_abs(c, -1)}
    )

    add_legend_patches(
        ax,
        [get_color(c) for c in supp_carriers],
        [get_label(c) for c in supp_carriers],
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
        [get_label(c) for c in cons_carriers],
        legend_kw={
            "bbox_to_anchor": (0.5, -pad),
            "ncol": 1,
            "title": "Consumption",
            **legend_kwargs,
        },
    )

    add_legend_semicircles(
        ax,
        [s * bus_size_factor * SEMICIRCLE_CORRECTION_FACTOR for s in legend_bus_sizes],
        [f"{s:g} {unit}" for s in legend_bus_sizes],
        patch_kw={"color": "#666"},
        legend_kw={"bbox_to_anchor": (0, 1), **legend_kwargs},
    )

    legend_branch_sizes = settings["branch_sizes"]
    if legend_branch_sizes is not None:
        add_legend_lines(
            ax,
            [s * branch_width_factor for s in legend_branch_sizes],
            [f"{s} {settings['unit']}" for s in legend_branch_sizes],
            patch_kw={"color": "#666"},
            legend_kw={"bbox_to_anchor": (0.25, 1), **legend_kwargs},
        )

    line_type_legend = ax.legend(
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
    ax.add_artist(line_type_legend)

    plt.show()
    return bus_size


def _electricity_capacities_cluster_vs_outside(n, bus_carriers, exclude_carriers,
                                               min_supply):
    """Optimised capacities [MW] of electricity-producing technologies,
    index: carrier, columns: 'outside', 'cluster'."""
    bus_carriers = list(bus_carriers)
    tables = [n.generators, n.links, n.storage_units, n.stores, n.lines, n.loads]

    # temporary column: inside the renewable cluster or not (by component name)
    for df in tables:
        df["cluster"] = np.where(
            df.index.str.contains("renewable cluster"), "cluster", "outside"
        )
    try:
        grouping = ["carrier", "cluster"]
        supply = n.statistics.supply(bus_carrier=bus_carriers, groupby=grouping,
                                     nice_names=False)
        caps = n.statistics.optimal_capacity(bus_carrier=bus_carriers, groupby=grouping,
                                             nice_names=False)
    finally:
        for df in tables:
            df.drop(columns="cluster", inplace=True)

    supply = supply.drop(list(exclude_carriers), level="carrier", errors="ignore")
    supply = supply.drop(["Line", "Load"], level="component", errors="ignore")
    producers = supply[supply > min_supply].index

    # per carrier (summed over component types), one column per side
    caps = (caps.reindex(producers)
                .groupby(level=["carrier", "cluster"]).sum()
                .unstack("cluster")
                .reindex(columns=["outside", "cluster"])
                .fillna(0.0))
    return caps


def plot_electricity_capacities_cluster_vs_outside(
    n,
    n_base=None,
    bus_carriers=("AC", "low voltage"),
    exclude_carriers=("AC", "DC", "electricity distribution grid"),
    min_supply=1,
    min_capacity=1,
    save_path=None,
    title=None,
):
    """Two bar charts of the optimised capacities of the technologies that
    produce electricity: left the ones outside the renewable clusters, right
    the ones inside (component name contains 'renewable cluster').

    If ``n_base`` is given, the left panel shows instead the difference of the
    capacities outside the renewable clusters with respect to ``n_base``
    (n - n_base, positive = more capacity in n). The right panel is unchanged.

    A technology produces electricity if it supplies more than ``min_supply``
    [MWh] over the year to a bus with carrier in ``bus_carriers``
    (n.statistics.supply). Transmission / distribution carriers in
    ``exclude_carriers`` are left out; note that the cluster grid connection
    links have the carrier of the node bus (AC), so they are left out too.
    Capacities are n.statistics.optimal_capacity, summed per carrier over
    all component types; only capacities (or absolute differences)
    > ``min_capacity`` [MW] are shown.

    ``title`` (optional str) is shown as the figure title above both panels.
    If ``save_path`` is given, the figure is saved there (e.g. as a PDF).

    Returns
    -------
    pd.DataFrame
        Capacities [MW], index: carrier, columns: 'outside', 'cluster'
        (+ 'outside_base' and 'outside_diff' if ``n_base`` is given).
    """
    caps = _electricity_capacities_cluster_vs_outside(
        n, bus_carriers, exclude_carriers, min_supply)

    if n_base is not None:
        base = _electricity_capacities_cluster_vs_outside(
            n_base, bus_carriers, exclude_carriers, min_supply)
        caps = caps.reindex(caps.index.union(base.index)).fillna(0.0)
        caps["outside_base"] = base["outside"].reindex(caps.index).fillna(0.0)
        caps["outside_diff"] = caps["outside"] - caps["outside_base"]

    def color(carrier):
        if carrier in label_to_colors:
            return label_to_colors[carrier]
        c = n.carriers.color.get(carrier, "") if "color" in n.carriers else ""
        return c if isinstance(c, str) and c else "#cccccc"

    def nice(carrier):
        name = n.carriers.nice_name.get(carrier, "") if "nice_name" in n.carriers else ""
        return name if isinstance(name, str) and name else carrier

    is_diff = n_base is not None
    left_col = "outside_diff" if is_diff else "outside"
    left_title = ("Outside the renewable clusters\n(difference vs. base)"
                  if is_diff else "Outside the renewable clusters")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, side, panel_title in zip(axes, [left_col, "cluster"],
                               [left_title, "Inside the renewable clusters"]):
        diff_panel = side == "outside_diff"
        col = caps[side]
        if diff_panel:
            values = col[col.abs() > min_capacity].sort_values(ascending=False)
        else:
            values = col[col > min_capacity].sort_values(ascending=False)
        if values.empty:
            ax.text(0.5, 0.5, "None", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="#777777")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(panel_title, fontsize=12, fontweight="bold")
            continue

        # GW if the largest capacity is above 10 GW, MW otherwise
        scale, unit = (1e3, "GW") if values.abs().max() > 1e4 else (1.0, "MW")
        x = np.arange(len(values))
        ax.bar(x, values.values / scale, color=[color(c) for c in values.index],
               edgecolor="white", linewidth=0.5, zorder=3)
        for xi, v in zip(x, values.values / scale):
            label = f"{v:+,.1f}" if diff_panel else f"{v:,.1f}"
            ax.text(xi, v, label, ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([nice(c) for c in values.index], rotation=45, ha="right",
                           fontsize=9)
        if diff_panel:
            ax.axhline(0, color="black", linewidth=0.8, zorder=4)
            ax.set_ylabel(f"Δ optimised capacity vs. base [{unit}]")
        else:
            ax.set_ylabel(f"Optimised capacity [{unit}]")
        ax.set_title(panel_title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()

    return caps


PICTURES_DIR = Path("/work/users/rofrug/projects/pypsa-eur/pictures")
PICTURES_DIR.mkdir(parents=True, exist_ok=True)
#%%
networks_folder=r"results/Iberic40_2035_industrial_clusters_all3hrerun/all/networks"
config_plotting = yaml.safe_load(Path("config/plotting.default.yaml").read_text())
regions=gdp.read_file(r'resources/Iberic40_2035_industrial_clusters/all/regions_onshore_base_s_40.geojson').set_index("name")


records = []
for path in sorted(Path(networks_folder).glob("*.nc")):
    if "-IC-" not in path.name:
        continue
    wc = parse_wildcards(path)
    records.append({**wc, "path": path})

df = pd.DataFrame(records)




n = pypsa.Network(str(df.loc[(df.CR == 0.1) & (df.BUYcap == 0.0) & (df.SELLcap == 0.0)  & (df.BOTHcap == 0.3), "path"].iloc[0]))

n_base=pypsa.Network(networks_folder+r"/base_s_40_lv1.0_3h_2035.nc")

el_caps_base = plot_electricity_capacities_cluster_vs_outside(n_base, title='Base', save_path=PICTURES_DIR / "ElCapacities_Iberian_Base.pdf")
el_caps = plot_electricity_capacities_cluster_vs_outside(n, n_base=n_base, title="CR = 0.1, capBUY = 0, capSELL = 0, capBOTH = 0.3", save_path=PICTURES_DIR / "ElCapacitiesDiff_Iberian_CR0p1_BOTHcap0p3.pdf")
el_caps = plot_electricity_capacities_cluster_vs_outside(n,  title="CR = 0.1, capBUY = 0, capSELL = 0, capBOTH = 0.3")


n = pypsa.Network(str(df.loc[(df.CR == 0.3) & (df.BUYcap == 0.0) & (df.SELLcap == 0.0)  & (df.BOTHcap == 0.1), "path"].iloc[0]))
el_caps = plot_electricity_capacities_cluster_vs_outside(n, n_base=n_base, title="CR = 0.3, capBUY = 0, capSELL = 0, capBOTH = 0.1", save_path=PICTURES_DIR / "ElCapacitiesDiff_Iberian_CR0p3_BOTHcap0p1.pdf")
el_caps = plot_electricity_capacities_cluster_vs_outside(n,  title="CR = 0.3, capBUY = 0, capSELL = 0, capBOTH = 0.1")

n = pypsa.Network(str(df.loc[(df.CR == 0.3) & (df.BUYcap == 0.0) & (df.SELLcap == 0.0)  & (df.BOTHcap == 0.5), "path"].iloc[0]))
el_caps = plot_electricity_capacities_cluster_vs_outside(n, n_base=n_base, title="CR = 0, capBUY = 0, capSELL = 0, capBOTH = 0", save_path=PICTURES_DIR / "ElCapacitiesDiff_Iberian_CR0_BOTHcap0p5.pdf")


#%%
cluster_components = (
    [("link", idx) for idx in n.links.index if ("cluster" in idx and "discharger" not in idx and "methanol renewable cluster" not in idx and n.links.loc[idx, "p_nom_opt"] > 0.1)]
    + [("generator", idx) for idx in n.generators.index if ("cluster" in idx and n.generators.loc[idx, "p_nom_opt"] > 0.1)]
)

load_durations, pnoms = plot_load_duration_curve(n, cluster_components, ["ES0 12"], small_capacity=None)

load_durations, pnoms = plot_load_duration_curve(n, cluster_components, ["ES0 4"], small_capacity=None)

self_supply_share, self_supply_avg = plot_cluster_self_supply_violin(n, "ES0 12")

self_supply_share, self_supply_avg = plot_cluster_self_supply_violin(n, "ES0 4")

price_curves = plot_cluster_price_duration_curves(n, "ES0 12", max_electricity_price=250, n_base=n_base,
                   #                               save_path=PICTURES_DIR / "PriceDurationCurves_ES0_11.pdf"
                                                    # max_hours=100
                                                    )

price_curves = plot_cluster_price_duration_curves(n, "ES0 4", max_electricity_price=250, n_base=n_base)

methanol_merit_order = plot_methanol_merit_order(n)

FT_merit_order = plot_FT_merit_order(n)

methanol_cost_breakdown = plot_cheapest_methanol_cost_breakdown(n, save_path=PICTURES_DIR / "MethanolCostBreakdown_Iberian.pdf")

FT_cost_breakdown = plot_cheapest_FT_cost_breakdown(n, save_path=PICTURES_DIR / "FTCostBreakdown_Iberian.pdf")

#%%

n = pypsa.Network(str(df.loc[(df.CR == 0.1) & (df.BUYcap == 0.0) & (df.SELLcap == 0.0)  & (df.BOTHcap == 0.3), "path"].iloc[0]))

self_supply_share, self_supply_avg = plot_cluster_self_supply_violin(n, "ES0 11")

self_supply_share, self_supply_avg = plot_cluster_self_supply_violin(n, "ES0 4")
#%%


el_balance_cluster = plot_electricity_balance_by_cluster(n, ["ES0 12"],start_date='1-1-2013', end_date='12-31-2013', n_base=n_base)

el_balance_cluster = plot_electricity_balance_by_cluster(n, ["ES0 11"],start_date='1-1-2013', end_date='12-31-2013', n_base=n_base)

el_balance_cluster = plot_electricity_balance_by_cluster(n, ["ES0 12"],start_date='1-15-2013', end_date='01-30-2013', n_base=n_base)

el_balance_cluster = plot_electricity_balance_by_cluster(n, ["ES0 12"],start_date='7-15-2013', end_date='7-30-2013', n_base=n_base)

el_balance_cluster = plot_electricity_balance_by_cluster(n, ["ES0 11"],start_date='1-15-2013', end_date='01-30-2013', n_base=n_base)

el_balance_cluster = plot_electricity_balance_by_cluster(n, ["ES0 11"],start_date='7-15-2013', end_date='7-30-2013', n_base=n_base)

#%%

cluster_el_balance_map = plot_cluster_electricity_balance_map(n, regions, config_plotting, label_to_colors, boundaries=[-15, 12, 35, 48])

n = pypsa.Network(str(df.loc[(df.CR == 0.4) & (df.BUYcap == 0.5) & (df.SELLcap == 0.0)  & (df.BOTHcap == 0.0), "path"].iloc[0]))
cluster_el_balance_map = plot_cluster_electricity_balance_map(n, regions, config_plotting, label_to_colors, boundaries=[-15, 12, 35, 48])

n = pypsa.Network(str(df.loc[(df.CR == 0.4) & (df.BUYcap == 0.0) & (df.SELLcap == 0.2)  & (df.BOTHcap == 0.0), "path"].iloc[0]))
cluster_el_balance_map = plot_cluster_electricity_balance_map(n, regions, config_plotting, label_to_colors, boundaries=[-15, 12, 35, 48])

n = pypsa.Network(str(df.loc[(df.CR == 0.4) & (df.BUYcap == 0.0) & (df.SELLcap == 0.0)  & (df.BOTHcap == 0.2), "path"].iloc[0]))
cluster_el_balance_map = plot_cluster_electricity_balance_map(n, regions, config_plotting, label_to_colors, boundaries=[-15, 12, 35, 48])

#%%

n = pypsa.Network(str(df.loc[(df.CR == 0.1) & (df.BUYcap == 0.0) & (df.SELLcap == 0.0)  & (df.BOTHcap == 0.5), "path"].iloc[0]))

# el_balance_cluster = plot_electricity_balance_by_cluster(n, ["ES0 12"],start_date='01-01-2013', end_date='03-31-2013')
# el_balance_cluster = plot_electricity_balance_by_cluster(n, ["ES0 12"],start_date='04-01-2013', end_date='06-30-2013')
# el_balance_cluster = plot_electricity_balance_by_cluster(n, ["ES0 12"],start_date='07-01-2013', end_date='09-30-2013')
el_balance_cluster = plot_electricity_balance_by_cluster(n, ["ES0 12"],start_date='12-01-2013', end_date='12-31-2013')

n = pypsa.Network(str(df.loc[(df.CR == 0.0) & (df.BUYcap == 0.0) & (df.SELLcap == 0.0)  & (df.BOTHcap == 0.0), "path"].iloc[0]))

# n.buses_t.marginal_price.loc[n.snapshots.month == 1, "ES0 12"].plot(figsize=(10, 5))
# n.buses_t.marginal_price.loc[n.snapshots.month == 2, "ES0 12"].plot(figsize=(10, 5))
# n.buses_t.marginal_price.loc[n.snapshots.month == 3, "ES0 12"].plot(figsize=(10, 5))
# n.buses_t.marginal_price.loc[n.snapshots.month == 4, "ES0 12"].plot(figsize=(10, 5))
# n.buses_t.marginal_price.loc[n.snapshots.month == 5, "ES0 12"].plot(figsize=(10, 5))
# n.buses_t.marginal_price.loc[n.snapshots.month == 6, "ES0 12"].plot(figsize=(10, 5))
# n.buses_t.marginal_price.loc[n.snapshots.month == 7, "ES0 12"].plot(figsize=(10, 5))
# n.buses_t.marginal_price.loc[n.snapshots.month == 8, "ES0 12"].plot(figsize=(10, 5))
# n.buses_t.marginal_price.loc[n.snapshots.month == 9, "ES0 12"].plot(figsize=(10, 5))
# n.buses_t.marginal_price.loc[n.snapshots.month == 10, "ES0 12"].plot(figsize=(10, 5))
# n.buses_t.marginal_price.loc[n.snapshots.month == 11, "ES0 12"].plot(figsize=(10, 5))
price_no_cluster=n.buses_t.marginal_price.loc[:, "ES0 12"]


n = pypsa.Network(str(df.loc[(df.CR == 0.0) & (df.BUYcap == 0.0) & (df.SELLcap == 0.0)  & (df.BOTHcap == 0.5), "path"].iloc[0]))
price_cluster=n.buses_t.marginal_price.loc[:, "ES0 12"]

diff=(price_cluster-price_no_cluster)
diff.plot(figsize=(10, 5))

# # cluster_curtailments = plot_cluster_curtailment(n, "ES0 31")
# cluster_curtailment_shares = plot_cluster_monthly_curtailment_share(n, "ES0 31")
# cluster_curtailment_energy = plot_cluster_monthly_curtailment_absolute(n, "ES0 31")


#%%


plot_methanol_production_el_price_colormap_map(n, regions, config_plotting, label_to_colors, boundaries=[-15, 12, 35, 48])
plot_methanol_production_onwind_cf_colormap(n, regions, config_plotting, label_to_colors, boundaries=[-15, 12, 35, 48])
plot_methanol_production_solar_cf_colormap(n, regions, config_plotting, label_to_colors, boundaries=[-15, 12, 35, 48])
plot_FT_production_map(n, config_plotting, label_to_colors, boundaries=[-15, 12, 35, 48])

methanol_prices_cluster, average_price_of_methanol_cluster = calculate_price_of_methanol_cluster(n)


#%%

networks_folder=r"results/Noridcs100_2035_industrial_clusters_all3hrerun/all/networks"
config_plotting = yaml.safe_load(Path("config/plotting.default.yaml").read_text())                        
regions=gdp.read_file(r'resources/Noridcs100_2035_industrial_clusters/all/regions_onshore_base_s_100.geojson').set_index("name")

records = []
for path in sorted(Path(networks_folder).glob("*.nc")):
    wc = parse_wildcards(path)
    records.append({**wc, "path": path})

df = pd.DataFrame(records)



n = pypsa.Network(str(df.loc[(df.CR == 0.1) & (df.BUYcap == 0.0) & (df.SELLcap == 0.0)  & (df.BOTHcap == 0.3), "path"].iloc[0]))

n_base=pypsa.Network(networks_folder+r"/base_s_100_lv1.0_3h_2035.nc")

el_caps_base = plot_electricity_capacities_cluster_vs_outside(n, title="Base", save_path=PICTURES_DIR / "ElCapacities_Nordics_Base.pdf")

el_caps = plot_electricity_capacities_cluster_vs_outside(n, n_base=n_base, title="CR = 0.1, capBUY = 0, capSELL = 0, capBOTH = 0.3", save_path=PICTURES_DIR / "ElCapacitiesDiff_Nordics_CR0p1_BOTHcap0p3.pdf")
el_caps = plot_electricity_capacities_cluster_vs_outside(n,  title="CR = 0.1, capBUY = 0, capSELL = 0, capBOTH = 0.3")


n = pypsa.Network(str(df.loc[(df.CR == 0.3) & (df.BUYcap == 0.0) & (df.SELLcap == 0.0)  & (df.BOTHcap == 0.1), "path"].iloc[0]))
el_caps = plot_electricity_capacities_cluster_vs_outside(n, n_base=n_base, title="CR = 0.3, capBUY = 0, capSELL = 0, capBOTH = 0.1", save_path=PICTURES_DIR / "ElCapacitiesDiff_Nordics_CR0p3_BOTHcap0p1.pdf")
el_caps = plot_electricity_capacities_cluster_vs_outside(n,  title="CR = 0.3, capBUY = 0, capSELL = 0, capBOTH = 0.1")

n = pypsa.Network(str(df.loc[(df.CR == 0.3) & (df.BUYcap == 0.0) & (df.SELLcap == 0.0)  & (df.BOTHcap == 0.5), "path"].iloc[0]))
el_caps = plot_electricity_capacities_cluster_vs_outside(n, n_base=n_base, title="CR = 0, capBUY = 0, capSELL = 0, capBOTH = 0", save_path=PICTURES_DIR / "ElCapacitiesDiff_Nordics_CR0_BOTHcap0p5.pdf")

#%%
cluster_components = (
    [("link", idx) for idx in n.links.index if ("cluster" in idx and "discharger" not in idx and "methanol renewable cluster" not in idx and n.links.loc[idx, "p_nom_opt"] > 0.1)]
    + [("generator", idx) for idx in n.generators.index if ("cluster" in idx and n.generators.loc[idx, "p_nom_opt"] > 0.1)]
)
methanol_merit_order = plot_methanol_merit_order(n)

FT_merit_order = plot_FT_merit_order(n)

el_balance_cluster = plot_electricity_balance_by_cluster(n, ["DK1 0"],start_date='1-1-2013', end_date='12-31-2013', n_base=n_base)

load_durations, pnoms = plot_load_duration_curve(n, cluster_components, ["DK1 0"])


self_supply_share, self_supply_avg = plot_cluster_self_supply_violin(n, "DK1 0")


price_curves = plot_cluster_price_duration_curves(n, "DK1 0", max_electricity_price=500, n_base=n_base,
                                                  save_path=PICTURES_DIR / "PriceDurationCurves_DK1_0.pdf")

methanol_cost_breakdown = plot_cheapest_methanol_cost_breakdown(n, save_path=PICTURES_DIR / "MethanolCostBreakdown_Nordics.pdf")

FT_cost_breakdown = plot_cheapest_FT_cost_breakdown(n, save_path=PICTURES_DIR / "FTCostBreakdown_Nordics.pdf")

#%%


el_balance_cluster = plot_electricity_balance_by_cluster(n, ["DE0 7"],start_date='1-1-2013', end_date='12-31-2013', n_base=n_base)

el_balance_cluster = plot_electricity_balance_by_cluster(n, ["DK1 0"],start_date='1-15-2013', end_date='01-30-2013', n_base=n_base)

el_balance_cluster = plot_electricity_balance_by_cluster(n, ["DK1 0"],start_date='7-15-2013', end_date='7-30-2013', n_base=n_base)

el_balance_cluster = plot_electricity_balance_by_cluster(n, ["DE0 7"],start_date='1-15-2013', end_date='01-30-2013', n_base=n_base)

el_balance_cluster = plot_electricity_balance_by_cluster(n, ["DE0 7"],start_date='7-15-2013', end_date='7-30-2013', n_base=n_base)


#%%
cluster_el_balance_map = plot_cluster_electricity_balance_map(n, regions, config_plotting, label_to_colors, boundaries=[-10, 28, 46, 73])

n = pypsa.Network(str(df.loc[(df.CR == 0.4) & (df.BUYcap == 0.5) & (df.SELLcap == 0.0)  & (df.BOTHcap == 0.0), "path"].iloc[0]))
cluster_el_balance_map = plot_cluster_electricity_balance_map(n, regions, config_plotting, label_to_colors, boundaries=[-10, 28, 46, 73])

n = pypsa.Network(str(df.loc[(df.CR == 0.4) & (df.BUYcap == 0.0) & (df.SELLcap == 0.2)  & (df.BOTHcap == 0.0), "path"].iloc[0]))
cluster_el_balance_map = plot_cluster_electricity_balance_map(n, regions, config_plotting, label_to_colors, boundaries=[-10, 28, 46, 73])

n = pypsa.Network(str(df.loc[(df.CR == 0.4) & (df.BUYcap == 0.0) & (df.SELLcap == 0.0)  & (df.BOTHcap == 0.2), "path"].iloc[0]))
cluster_el_balance_map = plot_cluster_electricity_balance_map(n, regions, config_plotting, label_to_colors, boundaries=[-10, 28, 46, 73])
#%%

n = pypsa.Network(str(df.loc[(df.CR == 0.2) & (df.BUYcap == 0.0) & (df.SELLcap == 0.0)  & (df.BOTHcap == 0.5), "path"].iloc[0]))

cluster_components = (
    [("link", idx) for idx in n.links.index if ("cluster" in idx  and "methanol renewable cluster" not in idx and n.links.loc[idx, "p_nom_opt"] > 0.1)]
    + [("generator", idx) for idx in n.generators.index if ("cluster" in idx and n.generators.loc[idx, "p_nom_opt"] > 0.1)]
)

el_balance_cluster = plot_electricity_balance_by_cluster(n, ["DK0 1"],start_date='01-07-2013', end_date='01-14-2013')

el_balance_cluster = plot_electricity_balance_by_cluster(n, ["DK0 1"],start_date='07-07-2013', end_date='07-14-2013')

cluster_curtailment_shares = plot_cluster_monthly_curtailment_share(n, "DK1 0")

load_durations, pnoms = plot_load_duration_curve(n, cluster_components, ["DK1 0"])


#%%

plot_methanol_production_el_price_colormap_map(n, regions, config_plotting, label_to_colors, boundaries=[-10, 28, 46, 73])
plot_methanol_production_onwind_cf_colormap(n, regions, config_plotting, label_to_colors, boundaries=[-10, 28, 46, 73])
plot_methanol_production_solar_cf_colormap(n, regions, config_plotting, label_to_colors, boundaries=[-10, 28, 46, 73])
plot_dac_captured_co2_onwind_cf_colormap(n, regions, config_plotting, label_to_colors, boundaries=[-10, 28, 46, 73])
plot_FT_production_map(n, config_plotting, label_to_colors, boundaries=[-10, 28, 46, 73])
methanol_prices_cluster, average_price_of_methanol_cluster = calculate_price_of_methanol_cluster(n)
# %%

networks_folder=r"results/Noridcs100_2035_industrial_clusters/all/networks"
config_plotting = yaml.safe_load(Path("config/plotting.default.yaml").read_text())                        
regions=gdp.read_file(r'resources/Noridcs100_2035_industrial_clusters/all/regions_onshore_base_s_100.geojson').set_index("name")

records = []
for path in sorted(Path(networks_folder).glob("*.nc")):
    wc = parse_wildcards(path)
    records.append({**wc, "path": path})

df = pd.DataFrame(records)

n = pypsa.Network(str(df.loc[(df.CR == 0.5) & (df.BUYcap == 0.0) & (df.SELLcap == 0.1)  & (df.BOTHcap == 0.0), "path"].iloc[0]))

#%%