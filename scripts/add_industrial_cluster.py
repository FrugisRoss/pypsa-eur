# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Adds a renewable cluster (bundled generators, storage and grid connection) at
every AC node of the network prepared by ``prepare_sector_network``.
"""

import logging

import pandas as pd
import pypsa

from scripts._helpers import configure_logging, set_scenario_config
from scripts.add_electricity import load_costs

logger = logging.getLogger(__name__)


def add_buses_of_renewable_cluster(n, nodes):
    for node in nodes:
        if not n.buses.index.str.contains(f"{node} renewable cluster").any():
            n.add(
                "Bus",
                name=f"{node} renewable cluster",
                v_nom=n.buses.at[node, "v_nom"],
                x=n.buses.at[node, "x"],
                y=n.buses.at[node, "y"],
                unit=n.buses.at[node, "unit"],
                location=n.buses.at[node, "location"],
                country=n.buses.at[node, "country"],
                carrier=n.buses.at[node, "carrier"],
                control=n.buses.at[node, "control"],
                substation_lv=n.buses.at[node, "substation_lv"],
                substation_off=n.buses.at[node, "substation_off"],
            )

        if not n.buses.index.str.contains(f"{node} battery renewable cluster").any():
            n.add(
                "Bus",
                name=f"{node} battery renewable cluster",
                v_nom=n.buses.at[f"{node} battery", "v_nom"],
                x=n.buses.at[f"{node} battery", "x"],
                y=n.buses.at[f"{node} battery", "y"],
                unit=n.buses.at[f"{node} battery", "unit"],
                location=n.buses.at[f"{node} battery", "location"],
                country=n.buses.at[f"{node} battery", "country"],
                carrier=n.buses.at[f"{node} battery", "carrier"],
                control=n.buses.at[f"{node} battery", "control"],
                substation_lv=n.buses.at[f"{node} battery", "substation_lv"],
                substation_off=n.buses.at[f"{node} battery", "substation_off"],
            )

        if not n.buses.index.str.contains(f"{node} H2 renewable cluster").any():
            n.add(
                "Bus",
                name=f"{node} H2 renewable cluster",
                v_nom=n.buses.at[f"{node} H2", "v_nom"],
                x=n.buses.at[f"{node} H2", "x"],
                y=n.buses.at[f"{node} H2", "y"],
                unit=n.buses.at[f"{node} H2", "unit"],
                location=n.buses.at[f"{node} H2", "location"],
                country=n.buses.at[f"{node} H2", "country"],
                carrier=n.buses.at[f"{node} H2", "carrier"],
                control=n.buses.at[f"{node} H2", "control"],
                substation_lv=n.buses.at[f"{node} H2", "substation_lv"],
                substation_off=n.buses.at[f"{node} H2", "substation_off"],
            )

        if not n.buses.index.str.contains(
            f"{node} methanol renewable cluster"
        ).any():
            n.add(
                "Bus",
                name=f"{node} methanol renewable cluster",
                v_nom=n.buses.at["EU methanol", "v_nom"],
                x=n.buses.at["EU methanol", "x"],
                y=n.buses.at["EU methanol", "y"],
                unit=n.buses.at["EU methanol", "unit"],
                location=n.buses.at["EU methanol", "location"],
                country=n.buses.at["EU methanol", "country"],
                carrier=n.buses.at["EU methanol", "carrier"],
                control=n.buses.at["EU methanol", "control"],
                substation_lv=n.buses.at["EU methanol", "substation_lv"],
                substation_off=n.buses.at["EU methanol", "substation_off"],
            )

        if not n.buses.index.str.contains(
            f"{node} co2 stored renewable cluster"
        ).any():
            n.add(
                "Bus",
                name=f"{node} co2 stored renewable cluster",
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

    return n


def add_links_of_renewable_cluster(
    n, nodes, cluster_cost_reduction, ongrid_buy, ongrid_sell, cluster_size, costs
):
    for node in nodes:
        ### H2 Electrolysis ###

        link_name = f"{node} H2 Electrolysis"

        n.add(
            "Link",
            name=link_name + " renewable cluster",
            bus0=n.links.at[link_name, "bus0"] + " renewable cluster",
            bus1=n.links.at[link_name, "bus1"] + " renewable cluster",
            p_nom_extendable=n.links.at[link_name, "p_nom_extendable"],
            carrier=n.links.at[link_name, "carrier"],
            efficiency=n.links.at[link_name, "efficiency"],
            capital_cost=n.links.at[link_name, "capital_cost"],
            marginal_cost=n.links.at[link_name, "marginal_cost"],
            lifetime=n.links.at[link_name, "lifetime"],
            reversed=False,
            overwrite=True,
        )

        ### Methanolization ###

        link_name = f"{node} methanolisation"
        cluster_methanol_bus_name = f"{node} methanol renewable cluster"

        n.add(
            "Link",
            name=link_name + " renewable cluster",
            bus0=n.links.at[link_name, "bus0"] + " renewable cluster",
            bus1=cluster_methanol_bus_name,
            bus2=n.links.at[link_name, "bus2"] + " renewable cluster",
            bus3=f"{node} co2 stored renewable cluster",
            p_nom_extendable=n.links.at[link_name, "p_nom_extendable"],
            p_min_pu=n.links.at[link_name, "p_min_pu"],
            carrier=n.links.at[link_name, "carrier"],
            efficiency=n.links.at[link_name, "efficiency"],
            efficiency2=n.links.at[link_name, "efficiency2"],
            efficiency3=n.links.at[link_name, "efficiency3"],
            capital_cost=n.links.at[link_name, "capital_cost"],
            marginal_cost=n.links.at[link_name, "marginal_cost"],
            lifetime=n.links.at[link_name, "lifetime"],
            reversed=False,
            overwrite=True,
        )

        n.add(
            "Link",
            name=f"{node} methanol renewable cluster",
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

        ### Fischer-Tropsch ###

        link_name = f"{node} Fischer-Tropsch"

        n.add(
            "Link",
            name=link_name + " renewable cluster",
            bus0=n.links.at[link_name, "bus0"] + " renewable cluster",
            bus1=n.links.at[link_name, "bus1"],
            bus2=f"{node} co2 stored renewable cluster",
            p_nom_extendable=n.links.at[link_name, "p_nom_extendable"],
            p_min_pu=n.links.at[link_name, "p_min_pu"],
            carrier=n.links.at[link_name, "carrier"],
            efficiency=n.links.at[link_name, "efficiency"],
            efficiency2=n.links.at[link_name, "efficiency2"],
            capital_cost=n.links.at[link_name, "capital_cost"],
            marginal_cost=n.links.at[link_name, "marginal_cost"],
            lifetime=n.links.at[link_name, "lifetime"],
            reversed=False,
            overwrite=True,
        )

        ### DAC ###

        # urban decentral chosen because the methanol plant is also connected
        # to the urban central district heating bus; revisit if that changes.
        link_name = f"{node} urban decentral DAC"

        n.add(
            "Link",
            name=link_name + " renewable cluster",
            bus0=n.links.at[link_name, "bus0"] + " renewable cluster",
            bus1=n.links.at[link_name, "bus1"],
            bus2=n.links.at[link_name, "bus2"],
            bus3=f"{node} co2 stored renewable cluster",
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

        if ongrid_sell:
            ### Electricity connection to grid ###

            link_name = f"{node} electricity renewable cluster"

            n.add(
                "Link",
                name=link_name,
                bus0=f"{node} renewable cluster",
                bus1=f"{node}",
                carrier=n.buses.at[f"{node}", "carrier"],
                p_nom_extendable=True,
                efficiency=1.0,
                capital_cost=costs.at["electricity grid connection", "capital_cost"],
                marginal_cost=0.0,
                reversed=False,
                overwrite=True,
            )
        
        if ongrid_buy:
            ### Electricity connection from grid ###

            link_name = f"{node} electricity renewable cluster back"
            n.add(
                "Link",
                name=link_name,
                bus0=f"{node}",
                bus1=f"{node} renewable cluster",
                carrier=n.buses.at[f"{node}", "carrier"],
                p_nom_extendable=True,
                efficiency=1.0,
                capital_cost=costs.at["electricity grid connection", "capital_cost"],
                marginal_cost=0.0,
                reversed=False,
                overwrite=True,
            )

    return n


def add_stores_of_renewable_cluster(n, nodes, cluster_cost_reduction):
    for node in nodes:
        link_name = f"{node} H2 Store"

        n.add(
            "Store",
            name=link_name + " renewable cluster",
            bus=n.stores.at[link_name, "bus"] + " renewable cluster",
            carrier=n.stores.at[link_name, "carrier"],
            e_nom_extendable=True,
            capital_cost=n.stores.at[link_name, "capital_cost"],
            marginal_cost=n.stores.at[link_name, "marginal_cost"],
            e_initial_per_period=n.stores.at[link_name, "e_initial_per_period"],
            e_cyclic=n.stores.at[link_name, "e_cyclic"],
            e_cyclic_per_period=n.stores.at[link_name, "e_cyclic_per_period"],
            lifetime=n.stores.at[link_name, "lifetime"],
            overwrite=True,
        )

        link_name = f"{node} battery charger"

        n.add(
            "Link",
            name=link_name + " renewable cluster",
            bus0=n.links.at[link_name, "bus0"] + " renewable cluster",
            bus1=n.links.at[link_name, "bus1"] + " renewable cluster",
            carrier=n.links.at[link_name, "carrier"],
            p_nom_extendable=True,
            efficiency=n.links.at[link_name, "efficiency"],
            capital_cost=n.links.at[link_name, "capital_cost"]
            * (1 - cluster_cost_reduction),
            marginal_cost=n.links.at[link_name, "marginal_cost"],
            lifetime=n.links.at[link_name, "lifetime"],
            reversed=False,
            overwrite=True,
        )

        link_name = f"{node} battery discharger"

        n.add(
            "Link",
            name=link_name + " renewable cluster",
            bus0=n.links.at[link_name, "bus0"] + " renewable cluster",
            bus1=n.links.at[link_name, "bus1"] + " renewable cluster",
            carrier=n.links.at[link_name, "carrier"],
            p_nom_extendable=True,
            efficiency=n.links.at[link_name, "efficiency"],
            capital_cost=n.links.at[link_name, "capital_cost"],
            marginal_cost=n.links.at[link_name, "marginal_cost"],
            lifetime=n.links.at[link_name, "lifetime"],
            reversed=True,
            overwrite=True,
        )

        link_name = f"{node} battery"

        n.add(
            "Store",
            name=link_name + " renewable cluster",
            bus=n.stores.at[link_name, "bus"] + " renewable cluster",
            carrier=n.stores.at[link_name, "carrier"],
            e_nom_extendable=True,
            capital_cost=n.stores.at[link_name, "capital_cost"]
            * (1 - cluster_cost_reduction),
            marginal_cost=n.stores.at[link_name, "marginal_cost"],
            e_initial_per_period=n.stores.at[link_name, "e_initial_per_period"],
            e_cyclic=n.stores.at[link_name, "e_cyclic"],
            e_cyclic_per_period=n.stores.at[link_name, "e_cyclic_per_period"],
            lifetime=n.stores.at[link_name, "lifetime"],
            overwrite=True,
        )

        link_name = f"{node} co2 stored"

        n.add(
            "Store",
            name=link_name + " renewable cluster",
            bus=f"{node} co2 stored renewable cluster",
            carrier=n.stores.at[link_name, "carrier"],
            e_nom_extendable=True,
            capital_cost=n.stores.at[link_name, "capital_cost"],
            marginal_cost=n.stores.at[link_name, "marginal_cost"],
            lifetime=n.stores.at[link_name, "lifetime"],
            e_initial_per_period=n.stores.at[link_name, "e_initial_per_period"],
            e_cyclic=n.stores.at[link_name, "e_cyclic"],
            e_cyclic_per_period=n.stores.at[link_name, "e_cyclic_per_period"],
            overwrite=True,
        )

    return n


def add_generators_of_renewable_cluster(
    n, cluster_size, cluster_cost_reduction, renewables, nodes_with_clusters, costs
):
    # dictionaries of dataframes by (node, renewable), sorting the generators
    # by average capacity factor in descending order, and the subset assigned
    # to the cluster once enough capacity has been collected.
    nodes_renewables_cf = {}
    clusters_generators = {}

    for node in nodes_with_clusters.copy():
        # if a node lacks enough capacity for any renewable in `renewables`,
        # the cluster cannot be built there and the node is skipped.
        insufficient_generators = False

        for renewable in renewables:
            nodes_renewables_cf[(node, renewable)] = pd.DataFrame(
                index=n.generators["p_nom_max"]
                .loc[n.generators.index.astype(str).str.contains(f"{node} .*{renewable}$")]
                .index,
                columns=["p_max_pu", "p_nom_max"],
            )

            clusters_generators[(node, renewable)] = pd.DataFrame()

            # highest mean p_max_pu determines the best generators available
            nodes_renewables_cf[(node, renewable)]["p_max_pu"] = (
                n.generators_t["p_max_pu"]
                .loc[
                    :,
                    n.generators_t["p_max_pu"]
                    .columns.astype(str)
                    .str.contains(f"{node} .*{renewable}$"),
                ]
                .mean()
            )
            nodes_renewables_cf[(node, renewable)]["p_nom_max"] = n.generators[
                "p_nom_max"
            ].loc[n.generators.index.astype(str).str.contains(f"{node} .*{renewable}$")]
            nodes_renewables_cf[(node, renewable)]["p_nom_min"] = n.generators[
                "p_nom_min"
            ].loc[n.generators.index.astype(str).str.contains(f"{node} .*{renewable}$")]

            nodes_renewables_cf[(node, renewable)] = nodes_renewables_cf[
                (node, renewable)
            ].sort_values("p_max_pu", ascending=False)
            # only what is not installed yet is made available for the cluster
            nodes_renewables_cf[(node, renewable)]["p_nom_avail"] = (
                nodes_renewables_cf[(node, renewable)]["p_nom_max"]
                - nodes_renewables_cf[(node, renewable)]["p_nom_min"]
            )

            number_gen = 0

            while (
                nodes_renewables_cf[(node, renewable)]
                .iloc[0 : number_gen + 1]["p_nom_avail"]
                .sum()
                <= cluster_size
            ):
                if number_gen >= len(nodes_renewables_cf[(node, renewable)]):
                    logger.info(
                        f"Not enough {renewable} capacity available at node "
                        f"{node} to reach cluster_size."
                    )
                    nodes_with_clusters.remove(node)
                    insufficient_generators = True
                    break

                number_gen += 1

            if insufficient_generators:
                break  # exits the for renewable loop -> goes to next node

            clusters_generators[(node, renewable)] = n.generators.loc[
                nodes_renewables_cf[(node, renewable)].index[0 : number_gen + 1]
            ]
            remaining_avail_capacity = (
                nodes_renewables_cf[(node, renewable)]
                .iloc[0 : number_gen + 1]["p_nom_avail"]
                .sum()
                - cluster_size
            )

            p_nom_available_last_cluster_gen = (
                cluster_size
                - clusters_generators[(node, renewable)]
                .loc[
                    clusters_generators[(node, renewable)].index[0:number_gen],
                    "p_nom_max",
                ]
                .sum()
                + clusters_generators[(node, renewable)]
                .loc[
                    clusters_generators[(node, renewable)].index[0:number_gen],
                    "p_nom_min",
                ]
                .sum()
            )

            clusters_generators[(node, renewable)].loc[
                clusters_generators[(node, renewable)].index[number_gen],
                "p_nom_max",
            ] = (
                clusters_generators[(node, renewable)].loc[
                    clusters_generators[(node, renewable)].index[number_gen],
                    "p_nom_min",
                ]
                + p_nom_available_last_cluster_gen
            )

            for idx in clusters_generators[(node, renewable)].index:
                ### Electricity generators ###

                p_nom_avail = (
                    clusters_generators[(node, renewable)].loc[idx].p_nom_max
                    - clusters_generators[(node, renewable)].loc[idx].p_nom_min
                )

                n.add(
                    "Generator",
                    name=clusters_generators[(node, renewable)].loc[idx].name
                    + " renewable cluster",
                    bus=clusters_generators[(node, renewable)].loc[idx].bus
                    + " renewable cluster",
                    carrier=clusters_generators[(node, renewable)].loc[idx].carrier,
                    p_nom_max=p_nom_avail,
                    p_max_pu=clusters_generators[(node, renewable)].loc[idx].p_max_pu,
                    marginal_cost=clusters_generators[(node, renewable)]
                    .loc[idx]
                    .marginal_cost,
                    capital_cost=(clusters_generators[(node, renewable)]
                    .loc[idx]
                    .capital_cost - costs.at["electricity grid connection", "capital_cost"])
                    * (1 - cluster_cost_reduction),
                    efficiency=clusters_generators[(node, renewable)]
                    .loc[idx]
                    .efficiency,
                    lifetime=clusters_generators[(node, renewable)].loc[idx].lifetime,
                    p_nom_extendable=True,
                    overwrite=True,
                )

                n.generators_t["p_max_pu"][
                    clusters_generators[(node, renewable)].loc[idx].name
                    + " renewable cluster"
                ] = n.generators_t["p_max_pu"][
                    clusters_generators[(node, renewable)].loc[idx].name
                ]

    return n


def add_renewable_cluster(
    n,
    nodes,
    cluster_size,
    cluster_cost_reduction,
    renewables,
    nodes_with_clusters,
    ongrid_buy,
    ongrid_sell,
    costs,
):
    n = add_buses_of_renewable_cluster(n, nodes)
    n = add_links_of_renewable_cluster(
        n, nodes, cluster_cost_reduction, ongrid_buy, ongrid_sell, cluster_size, costs
    )
    n = add_stores_of_renewable_cluster(n, nodes, cluster_cost_reduction)
    n = add_generators_of_renewable_cluster(
        n, cluster_size, cluster_cost_reduction, renewables, nodes_with_clusters, costs
    )

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_industrial_cluster",
            clusters="40",
            opts="",
            sector_opts="3h",
            planning_horizons=2035,
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)

    nyears = n.snapshot_weightings.generators.sum() / 8760

    costs = load_costs(
        snakemake.input.costs,
        snakemake.params.costs,
        nyears=nyears,
    )

    nodes = n.buses.loc[
        n.buses.index.str[:2].isin(snakemake.params.countries)
        & (n.buses["carrier"] == "AC")
    ].index.tolist()

    n = add_renewable_cluster(
        n,
        nodes,
        cluster_size=snakemake.params.cluster_size,
        cluster_cost_reduction=snakemake.params.cost_reduction,
        renewables=set(snakemake.params.renewables),
        nodes_with_clusters=nodes,
        ongrid_buy=snakemake.params.ongrid_buy,
        ongrid_sell=snakemake.params.ongrid_sell,
        costs=costs,
    )

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
