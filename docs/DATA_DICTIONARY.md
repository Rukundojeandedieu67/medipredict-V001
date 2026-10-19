# Data dictionary

## Reference CSVs

| File | Grain | Important fields |
|---|---|---|
| `dim_district.csv` | One row per district (30) | `adm2_pcode`, `district`, `province`, `lat`, `lon`, admissions/facility counts, joined blood-use totals, `data_source` |
| `dim_facility_osm.csv` | One row per deduplicated OSM facility (1,310) | `id`, `name_clean`, `tier`, `district`, `lat`, `lon`, `data_source` |
| `kigali_referral_nodes.csv` | One row per official Kigali referral hospital (4) | `node_id`, `facility_name`, district code, ownership/type, source and `data_source` |
| `district_pairs.csv` | Directed, non-self district pair (870) | `straight_km`, placeholder `est_road_km`, placeholder `est_transit_hours`, `data_source` |
| `district_distance_km.csv` | All directed district pairs including self (900) | Straight-line `straight_km`, `data_source` |
| `fact_hospitalisation.csv` | District x facility-type record (153) | `admissions`, nullable `avg_los_days`, `data_source` |
| `fact_blood_usage.csv` | District x facility x blood group (400) | `units`, `blood_group`, `data_source`; not used to set shares |
| `rbc_2024_collected_by_group_rcbt.csv` | RCBT x blood group (40) | `units_collected`, 2024, `source`, `data_source` |
| `rbc_2024_national_indicators.csv` | National indicator x year (11) | `indicator`, `value`, `table`, `source`, `data_source` |
| `rbc_2024_transfusion_indications.csv` | Reported indication (9) | RBC/platelet/FFP/cryo counts, total components, `source`, `data_source` |
| `blood_group_mix_comparison.csv` | Blood group (8) | Comparison fields; explicitly audit-only and never used by simulator/model |

## Simulated observations

| Field | Meaning |
|---|---|
| `timestamp` | UTC end of the daily or hourly observation period; the demand, receipts, and expiry in the row have occurred by this time |
| `node_id`, `node_name`, `node_type` | Unique simulated stock node (district blood bank by default; optional Ministry-listed Kigali referral hospital) |
| `district`, `district_code`, `province` | Simulated blood-bank location linked to the reference district |
| `blood_group` | Exact ABO/Rh group from the RBC 2024 collection mix |
| `stock_open_units` | Available stock after period-start expiry and deliveries, immediately before that period's simulated demand |
| `stock_units` | Remaining stock at the end of the period, after recorded demand is fulfilled and unmet demand is determined |
| `expiring_units_7d` | Remaining units in inventory batches expiring within the next seven days, known at the period-end timestamp |
| `demand_units` | Negative-binomial demand plus any separately logged synthetic surge units |
| `supply_received` | Deliveries during the period; includes opening stock on the first period |
| `expired_units` | Units expired at the start of the period under FEFO inventory tracking |
| `unmet_demand_units` | Positive demand not fulfilled from available stock; this is an outcome, not a generated label |
| `recent_surge_flag` | 1 when a synthetic surge event is logged at this node/group and timestamp, otherwise 0 |
| `lead_time_hours` | Delivered order's sampled lead time, or expected mode-weighted lead time when no delivery arrives |
| `delivery_mode` | `opening`, `road`, `drone`, or `none` |
| `data_source` | Always `simulated` in generated observation/event files |

`demand_events.csv` has one row per surge event and affected blood group, including event identifier, period-end timestamp, node, district, event type, added units, and `data_source`. Paired O+ and O- rows share the same event prefix but have distinct event IDs.

## Phase 2 derived rows

The pooled model uses one row per node, blood group, and prediction timestamp. Rolling demand/supply windows, days of cover, delivery recency, expiry exposure, weekday, district admissions weight, RBC 2024 group share, and group indicators use only data available at or before time t. `y_24h` and `y_48h` are labels from observed `unmet_demand_units > 0` strictly in rows t+1 through t+h; rows without a complete horizon are dropped. These model rows inherit `data_source=simulated` through their input readings.

## Database tables

The migration creates `districts`, `facilities`, `nodes`, `district_distance`, `blood_groups`, `ref_hospitalisation`, `ref_blood_usage`, `ref_rbc_2024`, `ref_blood_group_mix_audit`, `stock_readings`, `demand_events`, `transfers`, `predictions`, `recommendations`, and `decisions_audit`. Every table has a `data_source` enum column. Raw reference attributes are retained as JSONB where a fixed normalized schema would otherwise discard source fields.
