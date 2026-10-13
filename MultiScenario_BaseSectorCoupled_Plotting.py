#%%
import copy
import pypsa
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Patch, Polygon
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from pypsa.plot import add_legend_semicircles
from pypsa.statistics import get_transmission_carriers
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
    'Net Balance': "#00FF00",
    'onwind': "#0281d6",
    'Fischer-Tropsch': "#4e42f5",
    'Sabatier': '#ebb028',

    # Renewable cluster variants
    'methanolisation renewable cluster': '#EB28B0',
    'H2 Electrolysis renewable cluster': "#473BF5",
    'solar-hsat renewable cluster': "#870000",
    'solar renewable cluster': "#bff542",
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

    'grid connection (sell) renewable cluster': '#a69594',
    'grid connection (buy) renewable cluster': '#858ba8',
}


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
    return factor


def compute_co2_stored_balance(n, config_plotting):
    """
    Annual balance of the 'co2 stored' carrier per location and technology.

    Positive values are supply (CO2 captured into the stored-CO2 buses),
    negative values are consumption (sequestration, synthetic fuels, ...).
    CO2 transport between nodes is excluded.

    Returns
    -------
    bus_size : pd.Series indexed by (location, carrier), in the unit of
        config_plotting["plotting"]["balance_map"]["co2 stored"] (Mt).
    """
    carrier = "co2 stored"
    if carrier not in n.buses.carrier.unique():
        raise ValueError(f"Carrier '{carrier}' is not in the network.")

    unit_conversion = config_plotting["plotting"]["balance_map"][carrier]["unit_conversion"]

    pypsa.set_option("params.statistics.round", 8)
    pypsa.set_option("params.statistics.drop_zero", True)
    pypsa.set_option("params.statistics.nice_names", False)

    # energy_balance sums over time weighting every snapshot with
    # n.snapshot_weightings, so the result is in tCO2 per year.
    eb = n.statistics.energy_balance(
        bus_carrier=carrier,
        groupby=["bus", "carrier"],
    )

    # drop CO2 transport between nodes
    transmission_carriers = get_transmission_carriers(
        n, bus_carrier=carrier
    ).rename({"name": "carrier"})
    transmission_pairs = set(zip(
        transmission_carriers.get_level_values("component"),
        transmission_carriers.get_level_values("carrier"),
    ))
    is_transmission = pd.Series(
        list(zip(
            eb.index.get_level_values("component"),
            eb.index.get_level_values("carrier"),
        )),
        index=eb.index,
    ).isin(transmission_pairs)
    eb = eb[~is_transmission]

    # collapse all the CO2 buses of a node ('co2 stored', 'co2 stored DAC',
    # 'co2 stored renewable cluster', ...) onto the node itself
    location = (
        pd.Series(eb.index.get_level_values("bus"))
        .map(n.buses.location)
        .fillna("EU")
        .replace("", "EU")
        .values
    )

    bus_size = (
        eb.groupby([location, eb.index.get_level_values("carrier").values])
        .sum()
        .div(unit_conversion)
    )
    bus_size.index.names = ["location", "carrier"]
    bus_size = bus_size[bus_size.abs() > 1e-6]

    return bus_size


def compute_co2_sequestration_potential(n, config_plotting):
    """
    CO2 sequestration potential per location (e_nom_max of the
    'co2 sequestered' stores), in Mt/a. Locations with no (or unlimited)
    potential are returned as NaN.
    """
    unit_conversion = config_plotting["plotting"]["balance_map"]["co2 stored"]["unit_conversion"]

    co2_stores = n.stores.loc[n.stores.carrier == "co2 sequestered"]

    location = co2_stores.bus.map(n.buses.location)
    stripped = co2_stores.bus.str.replace(" co2 sequestered", "", regex=False).str.strip()
    location = location.where(location.notna() & (location != ""), stripped)

    storage_potential = (
        co2_stores.e_nom_max
        .replace(np.inf, np.nan)
        .groupby(location)
        .sum(min_count=1)
        .div(unit_conversion)
    )
    storage_potential[storage_potential.abs() < 1e-6] = np.nan

    return storage_potential


def plot_co2_stored_balance_maps(
    cases,
    config_plotting,
    label_to_colors,
    geo_scale=0.15,
    legend_sizes=None,
    figsize=(10, 6.5),
    save_path=None,
):
    """
    One figure with one 'co2 stored' balance map per case, side by side.

    Every node shows a split circle: the upper half is the CO2 supplied to
    the stored-CO2 buses, the lower half the CO2 consumed, both split by
    technology. The regions are coloured by the CO2 sequestration potential,
    on a colour scale shared by all the maps (single colorbar on the right).
    A single technology legend is placed below the maps.

    Parameters
    ----------
    cases : list of dict
        One dict per map with keys
        'n' (pypsa.Network), 'regions' (GeoDataFrame indexed by node name),
        'boundaries' ([lon_min, lon_max, lat_min, lat_max]) and 'title'.
    geo_scale : float
        Scaling of the circles: radius [deg] = sqrt(size [Mt]) * geo_scale.
    legend_sizes : list of float, optional
        Sizes shown in the circle-size legends. Defaults to the 'bus_sizes'
        of the 'co2 stored' balance map in the plotting config.
    save_path : str or Path, optional
        If given, the figure is saved there.
    """
    config = config_plotting["plotting"]
    settings = config["balance_map"]["co2 stored"]
    unit = settings["unit"]
    eu_location = config["eu_node_location"]

    if legend_sizes is None:
        legend_sizes = settings.get("bus_sizes", [])

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    bus_sizes = []
    regions_list = []
    for case in cases:
        n = case["n"]
        bus_sizes.append(compute_co2_stored_balance(n, config_plotting))

        regions = case["regions"].copy()
        # NOTE: no fillna here - nodes with no potential stay NaN
        # so geopandas skips them instead of drawing them as 0.
        regions["storage_potential"] = compute_co2_sequestration_potential(
            n, config_plotting
        ).reindex(regions.index)
        regions_list.append(regions)

    # colour scale shared by all the maps
    all_potentials = pd.concat([r.storage_potential for r in regions_list])
    vmin = all_potentials.min()  # NaNs ignored automatically
    vmax = all_potentials.max()
    if pd.isna(vmin):
        vmin, vmax = 0, 1
    elif vmin == vmax:
        vmax += 1

    orange_cmap = mcolors.LinearSegmentedColormap.from_list(
        "orange_scale",
        ["#fff2e0", "#e07b00", "#663500"],
    )
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    def get_color(n, c):
        if c in label_to_colors:
            return label_to_colors[c]
        color = n.carriers.color.get(c, "")
        return color if isinstance(color, str) and color != "" else "lightgrey"

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------
    crs = load_projection(copy.deepcopy(config))

    fig, axes = plt.subplots(
        1,
        len(cases),
        figsize=figsize,
        subplot_kw={"projection": crs},
        layout="constrained",
        squeeze=False,
    )
    axes = axes.ravel()

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

    carrier_colors = {}

    for ax, case, regions, bus_size in zip(axes, cases, regions_list, bus_sizes):
        n = case["n"]
        boundaries = case["boundaries"]

        ax.set_extent(boundaries, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
        regions.to_crs(crs.proj4_init).plot(
            ax=ax,
            column="storage_potential",
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

        area_correction = get_projected_area_factor(ax, boundaries, srid=4326)

        for loc, group in bus_size.groupby(level="location"):
            group = group.droplevel("location")

            if loc in n.buses.index and loc != "EU":
                x = float(n.buses.at[loc, "x"])
                y = float(n.buses.at[loc, "y"])
            else:
                x, y = eu_location["x"], eu_location["y"]
            x_proj, y_proj = ax.projection.transform_point(x, y, ccrs.PlateCarree())

            # upper half: supply (theta 180 -> 0), lower half: consumption (theta 180 -> 360)
            for values, direction in [
                (group[group > 0], -1),
                (group[group < 0].abs(), 1),
            ]:
                total = values.sum()
                if total <= 0:
                    continue

                r_corrected = np.sqrt(total) * geo_scale * area_correction
                values = values.sort_values(ascending=False)

                theta1 = 180
                for tech, value in values.items():
                    color = get_color(n, tech)
                    carrier_colors[tech] = color

                    theta2 = theta1 + direction * value / total * 180

                    angles = np.linspace(np.radians(theta1), np.radians(theta2), 100)
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

        ax.set_title(case.get("title", ""), fontsize=8)

        if len(legend_sizes) > 0:
            add_legend_semicircles(
                ax,
                [s * geo_scale**2 for s in legend_sizes],
                [f"{s} {unit}" for s in legend_sizes],
                patch_kw={"color": "#666"},
                legend_kw={
                    "bbox_to_anchor": (0, 1),
                    **size_legend_kwargs,
                },
            )

    # --------------------------------------------------
    # Colorbar (shared, on the right)
    # --------------------------------------------------
    sm = plt.cm.ScalarMappable(cmap=orange_cmap, norm=norm)
    cbar = fig.colorbar(
        sm,
        ax=axes,
        location="right",
        shrink=0.8,
        pad=0.02,
        aspect=40,
    )
    cbar.outline.set_edgecolor("None")
    cbar.set_label(f"CO₂ sequestration potential [{unit}CO₂/a]", fontsize=6)
    cbar.ax.tick_params(labelsize=6)

    # --------------------------------------------------
    # Technology legend (shared, below the maps)
    # --------------------------------------------------
    techs = sorted(carrier_colors)
    handles = [Patch(facecolor=carrier_colors[t], edgecolor="None") for t in techs]
    fig.legend(
        handles,
        techs,
        loc="outside lower center",
        ncol=min(len(techs), 4),
        frameon=False,
        fontsize=6,
        title="CO₂ stored balance (upper half: supply, lower half: consumption)",
        title_fontproperties={
            "weight": "bold",
            "size": 7,
        },
        handlelength=1.0,
        handleheight=1.0,
        handletextpad=0.4,
        labelspacing=0.25,
        columnspacing=1.5,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return bus_sizes, regions_list


#%%
config_plotting = yaml.safe_load(Path("config/plotting.default.yaml").read_text())

fn_nordics = r'results/Noridcs100_2035_industrial_clusters_all3h/all/networks/base_s_100__3h_2035.nc'
fn_iberian = r'results/Iberic40_2035_industrial_clusters_all3h/all/networks/base_s_40__3h_2035.nc'

n_nordics = pypsa.Network(fn_nordics)
n_iberian = pypsa.Network(fn_iberian)

regions_nordics = gpd.read_file(r'resources/Noridcs100_2035_industrial_clusters/all/regions_onshore_base_s_100.geojson').set_index("name")
regions_iberian = gpd.read_file(r'resources/Iberic40_2035_industrial_clusters/all/regions_onshore_base_s_40.geojson').set_index("name")

#%%
bus_sizes, regions_potential = plot_co2_stored_balance_maps(
    [
        {
            "n": n_nordics,
            "regions": regions_nordics,
            "boundaries": [-10, 28, 46, 73],
            "title": "Nordics",
        },
        {
            "n": n_iberian,
            "regions": regions_iberian,
            "boundaries": [-15, 12, 35, 48],
            "title": "Iberian Peninsula",
        },
    ],
    config_plotting,
    label_to_colors,
)
