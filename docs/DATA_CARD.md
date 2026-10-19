# Data card

## Scope

Phase 1 creates a reproducible district-level, daily inventory simulation for 30 Rwanda districts and eight blood groups. Simulated stock outcomes are not observed hospital data and must not be presented as such. No patient-level information is included.

## Provenance

| Dataset / field family | Source | Row-level `data_source` | Use |
|---|---|---|---|
| District codes, names, province, approximate location and joined counts in `dim_district.csv` | Prepared from Ministry of Health facility booklet, hospitalisation data and OSM; locations/counts retain source-specific limitations | `derived` | District weighting and geographic context |
| `dim_facility_osm.csv` | OpenStreetMap Rwanda facility snapshot (2026-10-07); OSM contributors | `real` | Facility context only; tags may be incomplete |
| `kigali_referral_nodes.csv` | Rwanda Ministry of Health Health Facilities Booklet 2023-2024; four Kigali entries marked `Referral Hospital` | `real` | Optional node expansion; disabled by default |
| `fact_hospitalisation.csv` | Supplied Rwanda hospitalisation extract; period not provided | `real` | District admissions weights and descriptive reference |
| `fact_blood_usage.csv` | Supplied blood-bank extract; reporting year/period not provided | `real` | Stored for audit and context; **not used for blood-group mix or simulation demand** |
| `district_pairs.csv` and `district_distance_km.csv` | Straight-line distance and estimated road/transit transformations | `derived` | Placeholder transport times; replace with routed times |
| `rbc_2024_collected_by_group_rcbt.csv` | RBC Blood Transfusion Division Annual Report 2024, Table 1.9 | `real` | The only group-share source used by the simulator |
| `rbc_2024_national_indicators.csv` | RBC Blood Transfusion Division Annual Report 2024, Tables 1.5, 1.6, 1.8 and 3.1 | `real`, except derived unmet-demand arithmetic (`derived`) | National calibration anchors |
| `rbc_2024_transfusion_indications.csv` | RBC Blood Transfusion Division Annual Report 2024, Table 4.3 | `real` | Clinical-use context; not an individual patient dataset |
| `blood_group_mix_comparison.csv` | Arithmetic comparison of the undated blood-bank extract with RBC 2024 collections | `derived` | Audit-only; prohibited from use as model or simulator input |
| `simulation_daily.csv` / `simulation_hourly.csv` | Seeded simulator | `simulated` | Synthetic inventory/demand; never a real-time observation |
| `demand_events.csv` | Seeded synthetic PPH/trauma surge process | `simulated` | Synthetic scenario events only |
| Future user/API entries | User or service submission | `entered` | Must remain distinguishable from real and simulated records |

Every CSV row produced in `data/reference/` or `data/generated/` has a `data_source` field. The database enforces the same four-value enum on all tables.

## Known limitations

- Hospitalisation and blood-use reporting periods are not present in the supplied extracts.
- District reference coordinates and distance-derived transit estimates are approximate. The road multiplier and 40 km/h assumption are placeholders, not routed travel estimates.
- RBC 2024 collections are used as a whole-blood-equivalent demand anchor only because no dated national demand stream is available. Collection units and transfused component units are not directly interchangeable.
- The simulator does not estimate real stock levels, forecast real stockouts, or prescribe clinical practice.
- If the optional Kigali referral-hospital nodes are enabled, each is assigned an equal share of its district's admissions weight together with that district's blood-bank node; no facility-level demand split is available.
- `transfusion.data` is the Taiwanese UCI donor dataset from 2007 and is not used by this system.
