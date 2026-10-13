#%%
import copy
import pypsa
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Patch, Polygon, Wedge
from matplotlib.legend_handler import HandlerBase
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
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


def plot_co2_stored_balance_maps(
    cases,
    config_plotting,
    label_to_colors,
    circle_scale=0.01,
    legend_sizes=None,
    fig_width=10,
    legend_ncol=1,
    save_path=None,
):
    """
    One figure with one 'co2 stored' balance map per case, side by side.

    Every node shows a split circle: the upper half is the CO2 supplied to
    the stored-CO2 buses, the lower half the CO2 consumed, both split by
    technology. The regions are coloured by the CO2 sequestration potential,
    on a colour scale shared by all the maps (single colorbar on the right).

    All the maps have the same size on the page: the boundaries of each case
    are enlarged (in one direction only) to a common height/width ratio. The
    circles are sized relative to the map width rather than in degrees, so
    the same amount of CO2 gives the same circle on every map, whatever its
    geographical scale. This is what allows a single circle-size legend,
    placed below the maps to the left of the technology legends (one column
    for the supply technologies and one for the consumption technologies).

    Parameters
    ----------
    cases : list of dict
        One dict per map with keys
        'n' (pypsa.Network), 'regions' (GeoDataFrame indexed by node name),
        'boundaries' ([lon_min, lon_max, lat_min, lat_max]) and 'title'.
    circle_scale : float
        Radius of a 1 Mt semicircle as a fraction of the map width
        (radius = sqrt(size [Mt]) * circle_scale * map width).
    legend_sizes : list of float, optional
        Sizes shown in the circle-size legend. Defaults to the 'bus_sizes'
        of the 'co2 stored' balance map in the plotting config.
    fig_width : float
        Figure width in inches; the height follows from the maps.
    legend_ncol : int
        Number of columns of each of the two technology legends.
    save_path : str or Path, optional
        If given, the figure is saved there.
    """
    config = config_plotting["plotting"]
    settings = config["balance_map"]["co2 stored"]
    unit = settings["unit"]
    eu_location = config["eu_node_location"]

    if legend_sizes is None:
        legend_sizes = settings.get("bus_sizes", [])
    legend_sizes = sorted(legend_sizes, reverse=True)

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    bus_sizes = []
    regions_list = []
    carrier_colors = {}
    for case in cases:
        n = case["n"]
        bus_size = compute_co2_stored_balance(n, config_plotting)
        bus_sizes.append(bus_size)

        for c in bus_size.index.unique("carrier"):
            color = label_to_colors.get(c, n.carriers.color.get(c, ""))
            if not isinstance(color, str) or color == "":
                color = "lightgrey"
            carrier_colors.setdefault(c, color)

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

    # split the technologies in supply and consumption; the ones appearing
    # on both sides go where they are larger (summed over all the maps)
    all_sizes = pd.concat(bus_sizes)
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
    crs = load_projection(copy.deepcopy(config))

    extents = [get_projected_extent(crs, case["boundaries"]) for case in cases]
    aspects = [(e[3] - e[2]) / (e[1] - e[0]) for e in extents]
    aspect = float(np.exp(np.mean(np.log(aspects))))
    extents = [pad_extent_to_aspect(e, aspect) for e in extents]

    # --------------------------------------------------
    # Figure layout (all lengths in inches)
    # --------------------------------------------------
    n_maps = len(cases)
    legend_fontsize = 6
    legend_title_fontsize = 7

    left, right, top, bottom = 0.1, 0.1, 0.3, 0.1
    wspace = 0.15
    cbar_pad, cbar_width, cbar_label_space = 0.15, 0.12, 0.55
    legend_pad = 0.2

    map_w = (
        fig_width - left - right - wspace * (n_maps - 1)
        - cbar_pad - cbar_width - cbar_label_space
    ) / n_maps
    map_h = map_w * aspect

    # radius of the legend semicircles in points (checked against the
    # circles actually drawn on the maps further down)
    legend_radii = [np.sqrt(s) * circle_scale * map_w * 72 for s in legend_sizes]
    max_radius = max(legend_radii, default=0)

    tech_rows = int(np.ceil(max(len(supp_carriers), len(cons_carriers)) / legend_ncol))
    tech_legend_h = (legend_title_fontsize * 1.6 + tech_rows * legend_fontsize * 1.5) / 72
    size_legend_h = (
        legend_title_fontsize * 1.6
        + len(legend_sizes) * (max_radius + legend_fontsize)
    ) / 72
    legend_h = max(tech_legend_h, size_legend_h) + 0.1

    fig_height = top + map_h + legend_pad + legend_h + bottom
    fig = plt.figure(figsize=(fig_width, fig_height))

    map_bottom = bottom + legend_h + legend_pad
    axes = [
        fig.add_axes(
            [
                (left + i * (map_w + wspace)) / fig_width,
                map_bottom / fig_height,
                map_w / fig_width,
                map_h / fig_height,
            ],
            projection=crs,
        )
        for i in range(n_maps)
    ]

    # --------------------------------------------------
    # Maps
    # --------------------------------------------------
    for ax, case, regions, bus_size, extent in zip(axes, cases, regions_list, bus_sizes, extents):
        n = case["n"]

        ax.set_extent(extent, crs=crs)
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
        ax.set_extent(extent, crs=crs)

        # radius of a 1 Mt semicircle in the units of the projection
        unit_radius = circle_scale * (extent[1] - extent[0])

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

        ax.set_title(case.get("title", ""), fontsize=8)

    # --------------------------------------------------
    # Check of the circle sizes: measure, on the rendered maps, the radius
    # in points of a 1 Mt semicircle and use it for the legend semicircles
    # --------------------------------------------------
    fig.canvas.draw()
    points_per_sqrt_mt = []
    for ax, extent in zip(axes, extents):
        x_pixels = ax.transData.transform([(0, 0), (circle_scale * (extent[1] - extent[0]), 0)])[:, 0]
        points_per_sqrt_mt.append((x_pixels[1] - x_pixels[0]) * 72 / fig.dpi)
    if not np.allclose(points_per_sqrt_mt, circle_scale * map_w * 72, rtol=0.01):
        print(
            "WARNING: the maps are not drawn at the expected size, radius of a "
            f"1 {unit} semicircle in points: expected {circle_scale * map_w * 72:.2f}, "
            f"drawn {np.round(points_per_sqrt_mt, 2)}"
        )
    legend_radii = [np.sqrt(s) * points_per_sqrt_mt[0] for s in legend_sizes]
    max_radius = max(legend_radii, default=0)

    # --------------------------------------------------
    # Colorbar (shared, on the right)
    # --------------------------------------------------
    cax = fig.add_axes(
        [
            (fig_width - right - cbar_label_space - cbar_width) / fig_width,
            (map_bottom + 0.1 * map_h) / fig_height,
            cbar_width / fig_width,
            0.8 * map_h / fig_height,
        ]
    )
    sm = plt.cm.ScalarMappable(cmap=orange_cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.outline.set_edgecolor("None")
    cbar.set_label(f"CO₂ sequestration potential [{unit}CO₂/a]", fontsize=6)
    cbar.ax.tick_params(labelsize=6)

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

    if len(legend_sizes) > 0:
        size_legend = lax.legend(
            [Wedge((0, 0), r, 0, 180, facecolor="#666", edgecolor="None") for r in legend_radii],
            [f"{s} {unit}" for s in legend_sizes],
            handler_map={Wedge: HandlerSemicircle()},
            bbox_to_anchor=(0, 1),
            title="Circle size",
            handlelength=2 * max_radius / legend_fontsize,
            handleheight=max_radius / legend_fontsize,
            labelspacing=0.6,
            **legend_kwargs,
        )
        lax.add_artist(size_legend)

    tech_legend_kwargs = {
        "ncol": legend_ncol,
        "handlelength": 1.0,
        "handleheight": 1.0,
        "handletextpad": 0.4,
        "labelspacing": 0.25,
        "columnspacing": 1.5,
        **legend_kwargs,
    }

    supply_legend = lax.legend(
        [Patch(facecolor=carrier_colors[t], edgecolor="None") for t in supp_carriers],
        supp_carriers,
        bbox_to_anchor=(0.2, 1),
        title="Supply (upper half)",
        **tech_legend_kwargs,
    )
    lax.add_artist(supply_legend)

    lax.legend(
        [Patch(facecolor=carrier_colors[t], edgecolor="None") for t in cons_carriers],
        cons_carriers,
        bbox_to_anchor=(0.5, 1),
        title="Consumption (lower half)",
        **tech_legend_kwargs,
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
            # Germany, Denmark, UK, Netherlands, Sweden and Norway
            "boundaries": [-9, 31.5, 47, 71.5],
            "title": "Nordics",
        },
        {
            "n": n_iberian,
            "regions": regions_iberian,
            # Spain and Portugal
            "boundaries": [-10.5, 5, 35.5, 44.5],
            "title": "Iberian Peninsula",
        },
    ],
    config_plotting,
    label_to_colors,
)

#%%