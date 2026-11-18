# Notes

## Vintage of existing capacities (`config/config.iberian_2035.industrial_clusters.yaml`)

*Checked 2026-11-16, powerplantmatching v0.8.0 (`pypsa-eur` conda env).*

With this config, existing capacities come from several sources with different vintages. Thermal plants are taken as of about 2024–2025. Solar and wind come mostly from GEM trackers dated February 2025, with an IRENA 2020 top-up that most likely adds nothing. Nothing is projected to 2035.

### How it works with `foresight: overnight`

Since the config runs overnight, the brownfield rule `add_existing_baseyear` is not used, so `grouping_years_power` and `grouping_years_heat` in `existing_capacities:` have no effect. Existing capacities enter in only two places:

**1. Conventional plants and hydro** come from rule `build_powerplants` → [scripts/build_powerplants.py:157](scripts/build_powerplants.py#L157). That calls `pm.powerplants(from_url=True)`, which downloads the matched powerplantmatching dataset. Its sources date from 2024–2025, for example GEM Oil & Gas August 2025, Coal July 2025, Hydro April 2025 and Beyond Fossil Fuels July 2025.

- The `powerplants_filter: (DateOut >= 2024 or DateOut != DateOut) ...` keeps every plant still running in 2024. It does **not** remove plants due to retire before 2035, so the fleet is roughly the one standing in 2024–2025.
- In the same script, biomass is replaced with OPSD renewable data from August 2020, for countries where OPSD has data (likely ES and PT).

**2. Solar and onshore/offshore wind** come from `estimate_renewable_capacities` in [scripts/add_electricity.py:1265-1282](scripts/add_electricity.py#L1265-L1282), in two steps:

- `from_gem: true` first places the units listed in GEM's solar and wind trackers. The cached files (`~/.local/share/powerplantmatching/data/in/`) are `Global-Solar-Power-Tracker-February-2025` and `Global-Wind-Power-Tracker-February-2025`, so this is roughly end-2024 capacity.
- Then `year: 2020` compares against IRENASTAT totals for **2020** and adds capacity only where GEM is below those totals ([scripts/add_electricity.py:963-985](scripts/add_electricity.py#L963-L985)). Spain and Portugal grew a lot after 2020, so the GEM numbers are probably already above IRENA 2020 and this step likely adds nothing. The `add_electricity` log prints the IRENA totals, so they can be compared with the GEM capacities.

### Summary

| Technology | Source | Vintage |
|---|---|---|
| Nuclear, coal, lignite, oil, gas, hydro, geothermal | powerplantmatching (GEM, BFF, …) | ~2024–2025 |
| Biomass (ES/PT) | OPSD | 2020 |
| Solar / onwind / offwind | GEM trackers + IRENA top-up | Feb 2025 trackers; top-up uses 2020 |

### Possible changes

- To make the renewables consistent, set `electricity.estimate_renewable_capacities.year: 2024`. The cached `IRENASTAT_capacities_2000-2024.csv` already covers that year.
- To drop plants that retire before the 2035 horizon, use `DateOut >= 2035` in `electricity.powerplants_filter`. The current filter keeps them in the 2035 network.

## Transmission grid behind `lv1.0` (Iberian and Nordic 2035 configs)

*Checked 2026-11-17, from the `resources/<prefix>/all/transmission_projects/` outputs of each run.*

`lv1.0` caps total transmission volume, Σ (s_nom or p_nom) × length over AC lines and DC links, at 1.0× the reference. The reference is computed in [scripts/prepare_network.py:152-180](scripts/prepare_network.py#L152-L180). Expansion is only switched on when the factor is above 1 or `opt`, so at `lv1.0` nothing is extendable.

The reference network is the **OSM base grid plus the transmission projects** added in `add_transmission_projects_and_dlr`, which runs on `base.nc` before clustering.

Both configs use the same settings:

- `base_network: osm-prebuilt` with `osm-prebuilt-version: 0.6`
- `transmission_projects`: tyndp2020, nep and manual all included
- `upgraded_lines` and `upgraded_links` skipped
- status limited to `under_construction`, `in_permitting` and `confirmed`
- `new_link_capacity: zero`

### References

- **OSM base grid (v0.6):** Xiong, B., Fioriti, D., Neumann, F., Riepin, I. & Brown, T. (2024). *Prebuilt Electricity Network for PyPSA-Eur based on OpenStreetMap Data* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14144752 (ODbL).
  - Methodology: https://doi.org/10.48550/arXiv.2408.17178.
  - [doc/data-base-network.rst](doc/data-base-network.rst) labels the citation "(0.5)", but the release notes and [rules/retrieve.smk:593](rules/retrieve.smk#L593) map record 14144752 to v0.6.
  - The record was published around November 2024. Check the Zenodo page for the exact OSM extraction date.
- **TYNDP 2020:** ENTSO-E project platform, https://tyndp2020-project-platform.azurewebsites.net/projectsheets. Data in [data/transmission_projects/tyndp2020/](data/transmission_projects/tyndp2020/).
- **NEP:** BNetzA, https://data.netzausbau.de/2037-2023/NEP/NEP_2037_2045_Bestaetigung.pdf. Data in [data/transmission_projects/nep/](data/transmission_projects/nep/).
- **Manual:** [data/transmission_projects/manual/new_links.csv](data/transmission_projects/manual/new_links.csv). Carried over from the old `links_tyndp.csv`, with TYNDP 2024 references in the tags.

Filtering is done in [scripts/build_transmission_projects.py:336-396](scripts/build_transmission_projects.py#L336-L396). A project is kept only if its whole geometry lies inside the modelled countries plus their offshore zones, and its status is in the list.

### What `new_link_capacity` does

The setting is applied in [scripts/build_transmission_projects.py:543-548](scripts/build_transmission_projects.py#L543-L548). It only affects **new DC links**. AC lines and upgraded links are not touched.

- `keep`: the link gets the source `p_nom` (0 if missing). It enters as existing, free capacity and adds to the `lv` reference volume.
- `zero`: the link is added with `p_nom = 0`. It only gains capacity if expansion is allowed (`lv` above 1.0 or `lvopt`), and then the optimiser has to decide to build it and pay for it.
- Under `lv1.0` with `zero`, new DC links sit at 0 MW and carry nothing.

### Iberian run (`Iberic40_2035_industrial_clusters_all3hrerun`; ES, PT)

- **New AC lines: 4, all from TYNDP 2020.**
  - TYNDP2020_0 to TYNDP2020_3, 380 kV double circuit, about 3396 MW each, `in_permitting`.
  - Lengths 131, 30, 140.2 and 140.2 km.
  - Project sheets [1](https://tyndp2020-project-platform.azurewebsites.net/projectsheets/transmission/1) and [4](https://tyndp2020-project-platform.azurewebsites.net/projectsheets/transmission/4).
  - Together they add about 1.5 TW·km to the reference.
- **NEP:** nothing; all its projects are German.
- **Manual list:** nothing.
  - The Biscay Gulf link (Gatica–Cubnezais) ends in France, which is not modelled.
  - APOLLO-LINK is `under_consideration`.
- **New DC links: none,** so `new_link_capacity` has no effect.

### Nordic run (`Noridcs100_2035_industrial_clusters_all3hrerun`; DK, DE, SE, NO, GB, NL)

**New AC lines: 27, adding about 10.7 TW·km.**

- 12 from TYNDP 2020, 380 kV double circuit, about 3396 MW each. Project ids 39, 78, 103, 124, 183, 207, 208, 258, 322, 1043 and 1063.
- 15 from the NEP: P43, P71, P84, P227, P402, P470, P476, P478, P485, P490, P540, P625, P627, P635 and P636.
  - All `confirmed`, build years 2030–2037.
  - About 5265 MW each (HTLS conductors).

**New DC links: 21, all set to 0 MW by `zero`.**

- 7 from TYNDP 2020, project ids 130, 132, 176, 235, 254 and 309. Three were merged with NEP entries:
  - `TYNDP2020_23_NEP_DC1`: Emden Ost–Osterath
  - `TYNDP2020_1_NEP_DC3`: Brunsbüttel–Großgartach
  - `TYNDP2020_2_NEP_DC4`: Wilster–Bergrheinfeld
- 13 from the NEP: DC5 SuedOstLink, DC20 SuedOstLink+, DC21, DC25, DC31, DC32, DC34, DC35, DC40, DC40plus, DC41, DC42 and DC42plus.
- 1 from the manual list: TYNDP2024_1229 Rhine-Main-Link (4 GW).

With `zero` + `lv1.0` the model effectively has **none of Germany's planned north–south HVDC corridors**, apart from what OSM already maps as built.

| | `zero` (current) | `keep` |
|---|---|---|
| Capacity of the 21 new DC links | 0 MW | Mostly 2000 MW each; Rhine-Main-Link 4000 MW; TYNDP 2020 links 700–2000 MW |
| Extra existing HVDC volume | 0 | ≈ 20 TW·km |
| Can carry power under `lv1.0` | No | Yes, as existing capacity at no cost |
| `lv1.0` reference volume | OSM + 10.7 TW·km of AC lines | OSM + ≈ 31 TW·km |

The ≈ 20 TW·km is source `p_nom` × length. 17.25 TW·km of it comes straight from the source files. The three merged TYNDP/NEP links were counted at the NEP's 2000 MW each, adding about 3.1 TW·km, because their source rows could not be matched directly.

### Open points

- Size relative to the whole grid: not yet computed (loading the network timed out). Compare the volume of `networks/base.nc` with `base_extended.nc`.
- Switching the Nordic run to `keep` means re-running from `build_transmission_projects` onwards. The Iberian run would not change.

## Cost and efficiency sources for methanol production

*Checked 2026-11-17, from `resources/Noridcs100_2035_industrial_clusters/all/costs_2030.csv`.*

All values come from **PyPSA technology-data v0.13.3** (`costs.version`). The 2035 Iberian and Nordic configs use cost year 2030, and `costs.overwrites` is empty, so they go into the model unchanged.

### Methanolisation (H₂ + CO₂ → methanol)

| Parameter | Value (2030) | Source |
|---|---|---|
| Investment | 703,726 EUR/MW_MeOH (2017 EUR) | Agora Energiewende (2018), Table 8 "Reference scenario" |
| FOM | 3.0 %/yr | Agora Energiewende (2018), section 6.3.2.1 |
| Lifetime | 20 yr | DEA, *Technology Data for Renewable Fuels* (04/2022), "Methanol to Power" sheet |
| H₂ input | 1.138 MWh_H2/MWh_MeOH (189 kg H₂/t MeOH) | DECHEMA (2017), p. 64 |
| Electricity input | 0.271 MWh_e/MWh_MeOH | DECHEMA (2017), p. 65 |
| CO₂ input | 0.248 tCO₂/MWh_MeOH | DECHEMA (2017), p. 66 |
| Heat output | 0.1 MWh_th/MWh_MeOH (2 GJ steam/t) | DECHEMA (2017), p. 65 |

- The investment cost falls from 703,726 EUR/MW_MeOH (2030) to 519,739 EUR/MW_MeOH (2050), even though technology-data describes it as "well developed technology, no significant learning expected". When quoting the number, say which cost year it is for.
- The lifetime is taken from the DEA "Methanol to Power" sheet, which is a different technology from methanol synthesis. Consider citing another source for it.

### Biomass-to-methanol (biomass gasification)

Source: **DEA, *Technology Data for Renewable Fuels***, datasheet "97 Methanol from biomass gasification" (2020 EUR).

| Parameter | 2030 | 2050 |
|---|---|---|
| Investment | 3,106 EUR/kW_MeOH | 1,553 EUR/kW_MeOH |
| FOM | 1.33 %/yr | 2.67 %/yr |
| VOM | 14.47 EUR/MWh_MeOH | 14.47 EUR/MWh_MeOH |
| Methanol efficiency | 0.61 | 0.65 |
| Electricity output | 0.02 MWh_e/MWh_th | 0.02 MWh_e/MWh_th |
| District heat output | 0.22 | 0.22 |
| Lifetime | 20 yr | 20 yr |

Not from the DEA:

- **Carbon split** (C in fuel 0.41, C stored 0.59, CO₂ stored 0.215 tCO₂/MWh_th): stoichiometric calculation following https://doi.org/10.1016/j.apenergy.2022.120016.
- **Capture rate 0.9** (CC variant, and also listed for methanolisation): assumption based on https://doi.org/10.1016/j.biombioe.2015.01.006.

### How the model uses them

- [scripts/prepare_sector_network.py:989-1034](scripts/prepare_sector_network.py#L989-L1034): biomass-to-methanol capital cost and VOM are **divided** by efficiency. The link's capacity is in MW of biomass input and the DEA costs are per kW of methanol, so the correct conversion is to **multiply** by efficiency, as the BtL link does. As written, the cost is overstated by 1/efficiency² ≈ 2.7× in 2030 (investment works out to ≈5,090 instead of ≈1,895 EUR per kW of biomass input). *Corrected 2026-11-17: an earlier version of this note said the division was the right conversion.*
- Methanolisation efficiencies are not read from the cost file but from the config (`MWh_MeOH_per_MWh_H2: 0.8787`, `MWh_MeOH_per_tCO2: 4.0321`, `MWh_MeOH_per_MWh_e: 3.6907`). These are the inverses of the DECHEMA values (1/1.138, 1/0.248, 1/0.271), so the DECHEMA citation still applies.
- [scripts/add_industrial_cluster.py:141-162](scripts/add_industrial_cluster.py#L141-L162): the renewable-cluster methanolisation links copy their parameters from the standard methanolisation links.

## Cost and efficiency sources for Fischer-Tropsch, BtL and electrobiofuels

*Checked 2026-11-17, same cost files and technology-data v0.13.3 as above. Both 2035 configs enable `biomass_to_liquid` and `electrobiofuels`; `biomass_to_liquid_cc` is off.*

The config also sets `custom_cost_fn: data/custom_costs.csv`, but that file does not exist and no script reads the setting, so it overrides nothing.

### Fischer-Tropsch (H₂ + CO₂ → liquid hydrocarbons)

| Parameter | 2030 | 2050 | Source |
|---|---|---|---|
| Investment | 703,726 EUR/MW_FT | 519,739 EUR/MW_FT | Agora Energiewende (2018), Table 8 "Reference scenario" (2017 EUR) |
| FOM | 3.0 %/yr | 3.0 %/yr | Agora Energiewende (2018), section 6.3.2.1 |
| VOM | 4.47 EUR/MWh_FT | | DEA, *Technology Data for Renewable Fuels*, sheet "102 Hydrogen to Jet" |
| Efficiency | 0.799 | | Agora Energiewende (2018), section 6.3.2.2 |
| H₂ input | 1.421 MWh_H2/MWh_FT | 1.327 MWh_H2/MWh_FT | DEA (2022), Hydrogen to Jet Fuel, Table 10 / p. 267 |
| Electricity input | 0.007 MWh_e/MWh_FT | | DEA (2022), Hydrogen to Jet Fuel, Table 10 / p. 267 |
| CO₂ input | 0.326 tCO₂/MWh_FT | 0.276 tCO₂/MWh_FT | DEA (2022), Hydrogen to Jet Fuel, Table 10 / p. 267 |
| Lifetime | 20 yr | | DEA (04/2022), "Methanol to Power" sheet |

- The investment, FOM and lifetime values are identical to methanolisation's and come from the same sources.
- The model's link ([scripts/prepare_sector_network.py:4690-4707](scripts/prepare_sector_network.py#L4690-L4707)) uses only `efficiency` (Agora, 0.799 MWh_FT/MWh_H2). The DEA `hydrogen-input` (1/1.421 ≈ 0.70) and `electricity-input` are **not used**. CO₂ input comes from the oil CO₂ intensity × efficiency, not from the DEA `carbondioxide-input`.

### Biomass-to-liquid (BtL, entrained-flow gasification + FT)

| Parameter | 2030 | 2050 | Source |
|---|---|---|---|
| Investment | 3,118 EUR/kW_th (2017 EUR) | 2,079 EUR/kW_th | Millinger et al. (2017), https://doi.org/10.1016/j.enpol.2017.05.013 |
| Efficiency | 0.383 | 0.45 | Millinger et al. (2017) |
| FOM | 2.67 %/yr | | DEA, sheet "85 Gasif. Ent. Flow FT, liq fu" |
| VOM | 1.13 EUR/MWh_FT | | DEA, sheet 85 |
| Lifetime | 25 yr | | DEA, sheet 85 |
| Carbon split | C in fuel 0.27, C stored 0.73, CO₂ stored 0.268 tCO₂/MWh_th | | Stoichiometric, Millinger et al. (2022), https://doi.org/10.1016/j.apenergy.2022.120016 |
| Capture rate | 0.9 | | Assumption based on Hannula (2015), https://doi.org/10.1016/j.biombioe.2015.01.006 |

- The "further description" column for investment and efficiency still names DEA sheet 85 ("Electricity Output" for efficiency), but the values come from Millinger et al. (2017). Cite Millinger for those two.
- In [scripts/prepare_sector_network.py:4224-4275](scripts/prepare_sector_network.py#L4224-L4275), capital cost and VOM are multiplied by efficiency, so they are per MW of biomass input.

### Electrobiofuels (BtL with added H₂)

Technology-data gives "combination of BtL and electrofuels" and "stoichiometric calculation" as sources. There is no separate report behind these rows.

| Parameter | 2030 | 2050 |
|---|---|---|
| efficiency-biomass | 1.322 | 1.328 |
| efficiency-hydrogen | 1.214 | 1.297 |
| efficiency-tot | 0.633 | 0.656 |
| C in fuel | 0.927 | 0.932 |
| Lifetime | 25 yr (fill value) | |

- The model ([scripts/prepare_sector_network.py:4277-4320](scripts/prepare_sector_network.py#L4277-L4320)) does **not** use the electrobiofuels investment, FOM or VOM rows. Capital and marginal costs are rebuilt as BtL cost + `BtL C stored` × Fischer-Tropsch cost. The cost sources are therefore those of BtL (Millinger 2017, DEA) and Fischer-Tropsch (Agora 2018, DEA).
- Technology-data has no `lifetime` row for electrobiofuels, so it takes `costs.fill_values.lifetime: 25`.
- The `investment` row (466,207 "EUR/kW_th" for 2030) is labelled in EUR/kW_th but is on a per-MW scale. It is unused, so this does not matter for the runs.
- The code marks the technology "experimental, use with caution".

## Cost and efficiency sources for hydrogen

*Checked 2026-11-17, same cost files and technology-data v0.13.3 as above.*

Switches in both 2035 configs: `SMR`, `SMR_cc`, `hydrogen_fuel_cell`, `hydrogen_turbine`, `hydrogen_underground_storage` and `ammonia` are on. `bioH2` (solid biomass to hydrogen), `H2_network`, `H2_retrofit` and `shipping_hydrogen_liquefaction` are off, so their cost rows are not used. `cc_fraction: 0.9` and `use_electrolysis_waste_heat: 0.25`.

### Production

| Technology | Parameter | 2030 | 2050 | Source |
|---|---|---|---|---|
| Electrolysis (AEC) | Investment | 1,500 EUR/kW_e | 1,000 EUR/kW_e | "private communications; IEA", *The Role of E-fuels in Decarbonising Transport* (2020 EUR) |
| | Efficiency | 0.622 | 0.699 | DEA, *Technology Data for Renewable Fuels*, sheet "86 AEC 100 MW" |
| | FOM | 4.0 %/yr | | DEA, sheet 86 |
| | Lifetime | 25 yr | | DEA, sheet 86 |
| SMR | Investment | 522,201 EUR/MW_CH4 (2015 EUR) | same | DEA, *Technology Data for Renewable Fuels*, table 3, p. 311 |
| | FOM | 5.0 %/yr | | DEA, same table |
| | Efficiency | 0.76 (LHV) | 0.76 | IEA (2020), "Global average levelised cost of hydrogen production by energy source and technology, 2019 and 2050" |
| | Lifetime | 30 yr | | IEA (2020), same chart |
| SMR CC | Investment | 605,753 EUR/MW_CH4 (2015 EUR) | same | DEA, table 3, p. 311 |
| | FOM | 5.0 %/yr | | DEA, same table |
| | Efficiency | 0.69 (LHV) | 0.69 | IEA (2020), same chart |
| | Lifetime | 30 yr | | IEA (2020), same chart |
| Ammonia cracker (NH₃ → H₂) | Investment | 1,123,945 EUR/MW_H2 (2015 EUR) | 558,309 EUR/MW_H2 | Ishimoto et al. (2020), table 6 |
| | FOM | 4.3 %/yr | | Ishimoto et al. (2020), table 7 |
| | Lifetime | 25 yr | | Ishimoto et al. (2020), table 7 |
| | Efficiency | 1/1.46 ≈ 0.685 MWh_H2/MWh_NH3 | | See note below |

Notes:

- **Electrolysis:** the link has no VOM. The investment row is the only one not from the DEA, and part of its source is "private communications", so the IEA e-fuels report is the citable part.
- **Electrolysis waste heat:** the DEA `efficiency-heat` (0.223) is **not used**. [scripts/prepare_sector_network.py:5398-5408](scripts/prepare_sector_network.py#L5398-L5408) sets heat output to (0.84 − efficiency) × `use_electrolysis_waste_heat`, i.e. (0.84 − 0.622) × 0.25 ≈ 0.055 MWh_th/MWh_e in 2030. The 0.84 is hard-coded.
- **SMR CC capture:** the link uses `cc_fraction: 0.9` from the config ([scripts/prepare_sector_network.py:2151-2167](scripts/prepare_sector_network.py#L2151-L2167)), not the `capture_rate` row. Both are 0.9; the IEA row notes a wide range of 54–90%.
- **SMR and SMR CC** have the same values in 2030 and 2050: no learning assumed.
- **Ammonia cracker efficiency** comes from `industry.MWh_NH3_per_MWh_H2_cracker: 1.46` in the config, which cites the TRACE repository (github.com/euronion/trace). Technology-data's `ammonia-input` row has the same value, sourced from ENGIE et al. (2020), *Ammonia to Green Hydrogen Feasibility Study*, Fig. 10. The cracker has no electricity input or VOM in the model.

### Re-electrification

| Technology | Parameter | 2030 | 2050 | Source |
|---|---|---|---|---|
| Fuel cell (LT-PEMFC) | Investment | 1,164 EUR/kW_e (2015 EUR) | 847 EUR/kW_e | DEA, *Technology Data for Generation of Electricity and District Heating*, sheet "12 LT-PEMFC CHP" |
| | Efficiency | 0.50 | 0.50 | DEA, sheet 12 |
| | FOM | 5.0 %/yr | | DEA, sheet 12 |
| | Lifetime | 10 yr | | DEA, sheet 12 |
| H₂ turbine | Investment | 460.6 EUR/kW_e (2015 EUR) | 435.8 EUR/kW_e | DEA, *Technology Data for Generation of Electricity and District Heating*, sheet "52 OCGT - Natural gas" |
| | Efficiency | 0.41 | 0.43 | DEA, sheet 52 |
| | FOM / VOM | 1.78 %/yr / 4.76 EUR/MWh | | DEA, sheet 52 |
| | Lifetime | 25 yr | | DEA, sheet 52 |

- The **H₂ turbine is a proxy**: it reuses the natural-gas OCGT data unchanged ([scripts/prepare_sector_network.py:1836-1854](scripts/prepare_sector_network.py#L1836-L1854), with a TODO to replace it). When describing it, say it is assumed equal to an OCGT.
- Fuel cell and turbine capital costs are multiplied by efficiency, converting per-kW_e figures to per kW of H₂ input.

### Storage

| Technology | Parameter | 2030 | 2050 | Source |
|---|---|---|---|---|
| Underground (salt caverns) | Investment | 2.13 EUR/kWh (2020 EUR) | 1.28 EUR/kWh | DEA, *Technology Data for Energy Storage*, sheet "151c Hydrogen Storage - Caverns" |
| | FOM / VOM | 0 / 0 | | DEA, sheet 151c |
| | Lifetime | 100 yr | | DEA, sheet 151c |
| Steel tank (type 1, incl. compressor) | Investment | 47.76 EUR/kWh (2020 EUR) | 22.33 EUR/kWh | DEA, *Technology Data for Energy Storage*, sheet "151a Hydrogen Storage - Tanks" |
| | FOM | 1.11 %/yr | | DEA, sheet 151a |
| | Lifetime | 30 yr | | DEA, sheet 151a |

- Caverns are added only at nodes with at least 2 TWh potential; every other node gets tanks ([scripts/prepare_sector_network.py:1856-1903](scripts/prepare_sector_network.py#L1856-L1903)).
- The H₂ store has no charging or discharging losses or power cost; the tank compressor cost is folded into the per-kWh investment.

### Cluster copies

The renewable- and point-source-cluster electrolysers copy their parameters from the standard `H2 Electrolysis` links ([scripts/add_industrial_cluster.py:120-137](scripts/add_industrial_cluster.py#L120-L137), [clusters/add_renewables_cluster.py:199](clusters/add_renewables_cluster.py#L199), [clusters/add_pointsource_cluster.py:362](clusters/add_pointsource_cluster.py#L362)), so they use the same sources.

## Cost and efficiency sources for direct air capture (DAC)

*Checked 2026-11-17, same cost files and technology-data v0.13.3 as above. `dac: true` in both 2035 configs.*

| Parameter | 2030 | 2050 | Source |
|---|---|---|---|
| Investment | 6,000,000 EUR/(tCO₂/h) (2020 EUR) | 4,000,000 EUR/(tCO₂/h) | DEA, *Technology Data for Carbon Capture, Transport and Storage*, sheet "403.a Direct air capture" |
| FOM | 4.95 %/yr | 4.95 %/yr | DEA, sheet 403.a |
| Lifetime | 20 yr | 20 yr | DEA, sheet 403.a |
| Electricity input | 0.4 MWh_e/tCO₂ | 0.4 | Beuttler et al. (2019), Climeworks low-temperature DAC |
| Heat input | 1.6 MWh_th/tCO₂ | 1.6 | Beuttler et al. (2019) |
| Compression electricity | 0.15 MWh_e/tCO₂ | 0.15 | DEA, sheet 403.a |
| Compression heat output | 0.2 MWh_th/tCO₂ | 0.2 | DEA, sheet 403.a |
| Heat output | 1.0 MWh/tCO₂ | 0.75 | DEA, sheet 403.a. **Not used by the model.** |

How the model builds the link ([scripts/prepare_sector_network.py:1204-1232](scripts/prepare_sector_network.py#L1204-L1232)):

- Electricity: 0.4 + 0.15 = **0.55 MWh_e/tCO₂**. Net heat: 1.6 − 0.2 = **1.4 MWh_th/tCO₂**.
- Capacity is in MW of electricity. Per MWh_e the link draws 1.4/0.55 ≈ 2.55 MWh_th from a heat bus, removes 1/0.55 ≈ 1.82 tCO₂ from `co2 atmosphere` and puts it into `co2 stored`.
- Capital cost is the per-(tCO₂/h) cost divided by 0.55. There is no VOM.
- One DAC link is added for each `urban central heat` and `services urban decentral heat` bus, so DAC only exists where those heat buses exist, and its heat must come from them. Technology-data notes the heat is meant to come from air-source heat pumps, but the model lets any supplier on that heat bus provide it.
- The renewable-cluster DAC copy ([scripts/add_industrial_cluster.py:200-226](scripts/add_industrial_cluster.py#L200-L226)) copies all parameters and only redirects stored CO₂ to the cluster's `co2 stored` bus.
- Technology-data gives Breyer et al. (2019) as an alternative with lower inputs (0.182 MWh_e and 1.102 MWh_th per tCO₂), but the model uses the Beuttler values.
- Beuttler et al. (2019) is written by Climeworks staff, so the energy figures are a manufacturer's own numbers.
- **DEA catalogue version** (checked 2026-11-17 in `inputs/technology_data_for_carbon_capture_transport_storage.xlsx` at technology-data tag v0.13.3): the Index sheet's version table has one entry, dated **2021-11-01**, "First published". The same sheet says "Cost data is in 2020€", which is where the `currency_year` 2020 in the cost file comes from.
- The DEA sheet 403.a has its own energy figures (2030: 2.0 MWh_th and 0.32 MWh_e per tCO₂), but technology-data replaces them with the Beuttler et al. values.

## Underground CO₂ sequestration potential per region

*Checked 2026-11-17 for the Iberian and Nordic 2035 configs.*

### Data source

- **CO2StoP**, the European CO₂ storage database of the European Commission's JRC, distributed through SETIS: https://setis.ec.europa.eu/european-co2-storage-database_en.
- Downloaded by rule `retrieve_co2stop` ([rules/retrieve.smk:423-440](rules/retrieve.smk#L423-L440)) as `co2jrc_openformats.zip` into `data/CO2JRC_OpenFormats/`.
- Files used ([rules/build_sector.smk:674-694](rules/build_sector.smk#L674-L694)):
  - `Hydrocarbon_Storage_Units.csv` + `StorageUnits_March13.kml`: saline aquifer storage units.
  - `Hydrocarbon_Traps*.csv` + `DaughterUnits_March13.kml`: oil and gas fields (traps).
- [doc/data-retrieval.rst:42-47](doc/data-retrieval.rst#L42-L47) gives the licence as "various".

### Processing chain

1. **`build_co2_sequestration_potentials`** ([scripts/build_co2_sequestration_potentials.py](scripts/build_co2_sequestration_potentials.py)) turns the raw CO2StoP tables and polygons into `resources/co2_sequestration_potentials.geojson`. The code was integrated from https://github.com/ericzhou571/Co2Storage (PyPSA-Eur PR #1606).
2. **`build_clustered_co2_sequestration_potentials`** ([scripts/build_clustered_co2_sequestration_potentials.py](scripts/build_clustered_co2_sequestration_potentials.py)) assigns the potential to model regions:
   - Attributes summed: `conservative estimate Mt`, `conservative estimate GAS Mt`, `conservative estimate OIL Mt`, `conservative estimate aquifer Mt`, i.e. the **conservative** CO2StoP estimates for aquifers plus oil and gas fields.
   - `include_onshore: false`: storage sites are overlaid with the **offshore regions only**. Onshore storage is excluded.
   - Sites with 3 Mt or less are dropped (`min_size: 3`), and so are regions whose total is 3 Mt or less.
   - Each site's potential is split between regions by area share.
3. **`prepare_sector_network`** ([scripts/prepare_sector_network.py:806-844](scripts/prepare_sector_network.py#L806-L844)):
   - Each region's potential is capped at `max_size: 25` Gt and divided by `years_of_storage: 25`. The result is the `e_nom_max` of that region's `co2 sequestered` store, so a region can inject at most 1/25 of its potential per modelled year.
   - Storage cost `co2_sequestration_cost: 30` EUR/tCO₂ and `co2_sequestration_lifetime: 50` years are config assumptions, not from CO2StoP.

### Total cap on top of the regional limits

- `co2_sequestration_potential` sets a model-wide cap in Mt/yr, applied as a global constraint in [scripts/solve_network.py:448-488](scripts/solve_network.py#L448-L488). For 2035 it is **33 Mt/yr (Nordic)** and **10 Mt/yr (Iberian)**. These values were set in the configs and are not from CO2StoP.
- No `seq` wildcard is used in `sector_opts`, so these config values are not overridden ([scripts/_helpers.py:823-825](scripts/_helpers.py#L823-L825)).

### Resulting potentials (from `resources/<prefix>/all/co2_sequestration_potential_base_s_<clusters>.csv`)

| Run | Regions with potential | Total (Mt) | Per year after /25 (Mt/yr) | By country (Mt) | Total cap 2035 |
|---|---|---|---|---|---|
| Nordic (`Noridcs100_..._all3hrerun`) | 23 of 100 | 77,949 | ≈ 3,118 | GB 54,581; DK 20,605; DE 1,978; NO 355; NL 227; SE 203 | 33 Mt/yr |
| Iberian (`Iberic40_..._all3hrerun`) | 6 of 40 | 5,491 | ≈ 220 | PT 5,363; ES 128 | 10 Mt/yr |

- No region reaches the 25 Gt `max_size` cap.
- The total caps (33 and 10 Mt/yr) are 1–5 % of the regional annual limits, so **the total cap is what binds**, not the CO2StoP data. CO2StoP mainly decides *where* CO₂ can be stored.
- Norway gets only 355 Mt, although the Norwegian shelf is usually reckoned among the largest storage resources in Europe. This probably reflects CO2StoP's coverage of Norway, so check it before relying on the regional split.

### Consequences

- Offshore regions are named after the onshore bus they connect to, so only **coastal** nodes get storage potential. Inland nodes get 0.
- With `co2_network: false`, CO₂ captured at an inland node cannot be moved to a coastal store. It can only be used there (e.g. methanolisation, Fischer-Tropsch). Venting is not possible either, because `co2_vent: false`.

## Which fuel processes also produce heat

*Checked 2026-11-17 in [scripts/prepare_sector_network.py:5340-5397](scripts/prepare_sector_network.py#L5340-L5397). All waste heat goes to `urban central heat`, so only at nodes that have district heating.*

| Technology | Heat in the model? | Amount (2030) |
|---|---|---|
| Methanolisation | **Yes** (`use_methanolisation_waste_heat: 0.25`) | DECHEMA heat-output / H₂ input × 0.25 = 0.1/1.138 × 0.25 ≈ **0.022 MWh_th per MWh_H2** (0.025 per MWh_MeOH) |
| Fischer-Tropsch | **Yes** (`use_fischer_tropsch_waste_heat: 0.25`) | (0.95 − 0.799) × 0.25 ≈ **0.038 MWh_th per MWh_H2**. The 0.95 is hard-coded, with no source. |
| Biomass-to-methanol | **No** | The DEA sheet gives 0.22 district heat and 0.02 electricity per unit of biomass, but the link has no heat or electricity output. |
| Biomass-to-liquid | **No** | No heat row in the cost file, no heat bus on the link. |
| Electrobiofuels | **No** | No heat bus on the link. |

- The **renewable-cluster copies** of methanolisation and Fischer-Tropsch in [scripts/add_industrial_cluster.py](scripts/add_industrial_cluster.py) do not copy the heat bus (`bus4` / `bus3`), so cluster plants produce no heat. This contradicts the comment at [scripts/add_industrial_cluster.py:202-203](scripts/add_industrial_cluster.py#L202-L203), which says the cluster methanol plant is connected to urban central heating. The older scripts in `clusters/` do copy `bus4`.

## References (BibTeX) for the methanol, FT, BtL, electrobiofuels, hydrogen and DAC sources

The journal entries were checked in Scite and their metadata comes from it. The reports were written from memory: check the fields marked `% check`.

```bibtex
@techreport{agora2018syntheticfuels,
  author      = {{Agora Verkehrswende} and {Agora Energiewende} and {Frontier Economics}},
  title       = {The Future Cost of Electricity-Based Synthetic Fuels},
  institution = {Agora Verkehrswende, Agora Energiewende and Frontier Economics},
  address     = {Berlin},
  year        = {2018},
  url         = {https://www.agora-energiewende.de/en/publications/the-future-cost-of-electricity-based-synthetic-fuels-1/}
}

@techreport{dechema2017lowcarbon,
  author      = {Bazzanella, Alexis Michael and Ausfelder, Florian},
  title       = {Low carbon energy and feedstock for the {European} chemical industry},
  institution = {DECHEMA Gesellschaft f{\"u}r Chemische Technik und Biotechnologie e.V.},
  address     = {Frankfurt am Main},
  year        = {2017},
  isbn        = {978-3-89746-196-2}, % check
  url         = {https://dechema.de/dechema_media/Downloads/Positionspapiere/Technology_study_Low_carbon_energy_and_feedstock_for_the_European_chemical_industry.pdf}
}

@techreport{dea2022renewablefuels,
  author      = {{Danish Energy Agency} and {Energinet}},
  title       = {Technology Data for Renewable Fuels},
  institution = {Danish Energy Agency},
  address     = {Copenhagen},
  year        = {2022}, % check: version used by technology-data v0.13.3
  url         = {https://ens.dk/en/our-services/projections-and-models/technology-data/technology-data-renewable-fuels} % check: ens.dk has moved pages
}

@article{millinger2017competitiveness,
  author  = {Millinger, Markus and Ponitka, Jens and Arendt, Oliver and Thr{\"a}n, Daniela},
  title   = {Competitiveness of advanced and conventional biofuels: Results from least-cost modelling of biofuel competition in {Germany}},
  journal = {Energy Policy},
  year    = {2017},
  volume  = {107},
  pages   = {394--402},
  doi     = {10.1016/j.enpol.2017.05.013}
}

@article{millinger2022biofuelmandates,
  author  = {Millinger, Markus and Reichenberg, Lina and Hedenus, Fredrik and Berndes, G{\"o}ran and Zeyen, Elisabeth and Brown, Tom},
  title   = {Are biofuel mandates cost-effective? -- An analysis of transport fuels and biomass usage to achieve emissions targets in the {European} energy system},
  journal = {Applied Energy},
  year    = {2022},
  volume  = {326},
  pages   = {120016},
  doi     = {10.1016/j.apenergy.2022.120016}
}

@article{hannula2015coproduction,
  author  = {Hannula, Ilkka},
  title   = {Co-production of synthetic fuels and district heat from biomass residues, carbon dioxide and electricity: Performance and cost analysis},
  journal = {Biomass and Bioenergy},
  year    = {2015},
  volume  = {74},
  pages   = {26--46},
  doi     = {10.1016/j.biombioe.2015.01.006}
}

@techreport{dea2016electricityDH,
  author      = {{Danish Energy Agency} and {Energinet}},
  title       = {Technology Data for Generation of Electricity and District Heating},
  institution = {Danish Energy Agency},
  address     = {Copenhagen},
  year        = {2016}, % check: catalogue is updated regularly; use the release technology-data v0.13.3 reads
  url         = {https://ens.dk/en/our-services/projections-and-models/technology-data/technology-data-generation-electricity-and} % check
}

@techreport{dea2018energystorage,
  author      = {{Danish Energy Agency} and {Energinet}},
  title       = {Technology Data for Energy Storage},
  institution = {Danish Energy Agency},
  address     = {Copenhagen},
  year        = {2018}, % check: catalogue is updated regularly; use the release technology-data v0.13.3 reads
  url         = {https://ens.dk/en/our-services/projections-and-models/technology-data/technology-data-energy-storage} % check
}

@misc{iea2020lcoh,
  author       = {{International Energy Agency}},
  title        = {Global average levelised cost of hydrogen production by energy source and technology, 2019 and 2050},
  howpublished = {Chart, IEA, Paris},
  year         = {2020},
  url          = {https://www.iea.org/data-and-statistics/charts/global-average-levelised-cost-of-hydrogen-production-by-energy-source-and-technology-2019-and-2050}
}

@techreport{iea2023efuels,
  author      = {{International Energy Agency}},
  title       = {The Role of E-fuels in Decarbonising Transport},
  institution = {International Energy Agency},
  address     = {Paris},
  year        = {2023}, % check
  url         = {https://iea.blob.core.windows.net/assets/9e0c82d4-06d2-496b-9542-f184ba803645/TheRoleofE-fuelsinDecarbonisingTransport.pdf}
}

@article{ishimoto2020hydrogen,
  author  = {Ishimoto, Yuki and Voldsund, Mari and Neks{\aa}, Petter and Roussanaly, Simon and Berstad, David and Gardarsdottir, Stefania},
  title   = {Large-scale production and transport of hydrogen from {Norway} to {Europe} and {Japan}: Value chain analysis and comparison of liquid hydrogen and ammonia as energy carriers},
  journal = {International Journal of Hydrogen Energy},
  year    = {2020},
  volume  = {45},
  number  = {58},
  pages   = {32865--32883},
  doi     = {10.1016/j.ijhydene.2020.09.017}
}

@techreport{engie2020ammoniatohydrogen,
  author      = {{ENGIE} and {Ecuity} and {Siemens} and {STFC}}, % check author list
  title       = {Ammonia to Green Hydrogen Project: Feasibility Study},
  institution = {UK Department for Business, Energy and Industrial Strategy},
  address     = {London},
  year        = {2020},
  url         = {https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/880826/HS420_-_Ecuity_-_Ammonia_to_Green_Hydrogen.pdf}
}

@techreport{dea2021ccts,
  author      = {{Danish Energy Agency} and {Energinet}},
  title       = {Technology Data for Carbon Capture, Transport and Storage},
  institution = {Danish Energy Agency},
  address     = {Copenhagen},
  year        = {2021},
  month       = nov,
  url         = {https://ens.dk/en/our-services/projections-and-models/technology-data/technology-data-carbon-capture-transport-and} % check
}

@techreport{poulsen2014co2stop,
  author      = {Poulsen, Niels and Holloway, Sam and Neele, Filip and Smith, Nichola A. and Kirk, Karen}, % check author list
  title       = {{CO2StoP} Final Report: Assessment of {CO2} storage potential in {Europe}},
  institution = {European Commission, Joint Research Centre},
  year        = {2014}, % check
  note        = {Project ENER/C1/154-2011-SI2.611598}, % check
  url         = {https://setis.ec.europa.eu/european-co2-storage-database_en}
}

@article{beuttler2019dac,
  author  = {Beuttler, Christoph and Charles, Louise and Wurzbacher, Jan Andre},
  title   = {The Role of Direct Air Capture in Mitigation of Anthropogenic Greenhouse Gas Emissions},
  journal = {Frontiers in Climate},
  year    = {2019},
  volume  = {1},
  doi     = {10.3389/fclim.2019.00010}
}
```
