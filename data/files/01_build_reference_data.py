"""
MediPredict - Step 1: clean raw sources and build reference tables.

Inputs  (IN_DIR):  health_facilities.gpkg, health-facilities_booklet-2023-2024.xlsx,
                   hospitalisation.csv, blood-bank.csv
Outputs (OUT_DIR): dim_district.csv, dim_facility_osm.csv, fact_blood_usage.csv,
                   fact_hospitalisation.csv, district_distance_km.csv,
                   district_pairs.csv, data_quality_report.md
Run:  python 01_build_reference_data.py [IN_DIR] [OUT_DIR]
"""
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

IN = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "processed")
OUT.mkdir(parents=True, exist_ok=True)
report = []

PROVINCE_SHORT = {"Kigali City": "Kigali City", "Western Province": "West",
                  "Southern Province": "South", "Eastern Province": "East",
                  "Northern Province": "North"}

def norm_province(s):
    s = s.astype(str).str.strip()
    return s.replace({"City of Kigali": "Kigali City"})

def norm_district(s):
    return s.astype(str).str.strip().str.title()

# ---------------------------------------------------------------- OSM layer
g = gpd.read_file(IN / "health_facilities.gpkg")
g["district"] = norm_district(g["adm2_name"])
g["province"] = g["adm1_name"].map(PROVINCE_SHORT)
g["name_clean"] = g["name_latin"].fillna(g["name"]).str.strip()
# polygons -> representative point so every feature has lat/lon
pts = g.geometry.representative_point()
g["lon"], g["lat"] = pts.x, pts.y
g["osm_kind"] = g["id"].str.split("/").str[0]

def tier(row):
    n = row["name_clean"].lower() if isinstance(row["name_clean"], str) else ""
    if row["amenity"] == "pharmacy" or "pharmac" in n or "farumasi" in n:
        return "pharmacy"
    if re.search(r"hospital|h[oô]pital|ibitaro|\bch[uk]\b|\bchu\b", n):
        return "hospital"
    if re.search(r"health post|poste de sant", n):
        return "health_post"
    if re.search(r"health cent|centre de sant|\bcs\b|\bhc\b", n):
        return "health_center"
    if "clinic" in n or row["amenity"] == "clinic":
        return "clinic"
    return "other"

g["tier"] = g.apply(tier, axis=1)
n_raw = len(g)
# same feature mapped as node AND polygon: keep one per (name, district), prefer node
g["_rank"] = g["osm_kind"].map({"node": 0, "way": 1, "relation": 2})
g = (g.sort_values("_rank")
       .assign(_key=lambda d: d["name_clean"].str.lower().fillna(d["id"]))
       .drop_duplicates(["_key", "district"]).drop(columns=["_rank", "_key"]))
report.append(f"OSM features: {n_raw} raw -> {len(g)} after node/polygon de-duplication "
              f"({n_raw-len(g)} removed).")
fac = g[["id", "name_clean", "amenity", "healthcare", "tier", "province", "district",
         "adm2_pcode", "adm3_name", "lat", "lon"]].rename(columns={"adm3_name": "sector"})
fac.to_csv(OUT / "dim_facility_osm.csv", index=False)
report.append("OSM tier counts: " + fac["tier"].value_counts().to_dict().__repr__())

# ------------------------------------------------------------ MoH booklet
hf = pd.read_excel(IN / "health-facilities_booklet-2023-2024.xlsx")
hf.columns = ["province", "district", "sector", "facility_name", "level", "ownership", "type"]
hf["province"] = norm_province(hf["province"]).replace(
    {"South": "South", "North": "North", "West": "West", "East": "East"})
hf["district"] = norm_district(hf["district"])
hf["type"] = hf["type"].str.strip()
bad = sorted(set(hf["district"]) - set(fac["district"]))
report.append(f"Booklet: {len(hf)} rows, districts after normalisation: {hf['district'].nunique()}, "
              f"unmatched to official list: {bad}")
HOSP_TYPES = ["District Hospital", "Referral Hospital", "Provincial Hospital",
              "Level Two Teaching Hospital", "Private Hospital", "Specialized Hospital"]
hf_district = (hf.assign(is_hospital=hf["type"].isin(HOSP_TYPES),
                         is_hc=hf["type"].isin(["Health Center", "Medicalized health Center"]),
                         is_hp=hf["type"].eq("Health Post"))
                 .groupby("district")
                 .agg(moh_facilities=("facility_name", "size"), moh_hospitals=("is_hospital", "sum"),
                      moh_health_centers=("is_hc", "sum"), moh_health_posts=("is_hp", "sum")))

# ---------------------------------------------------------- hospitalisation
hos = pd.read_csv(IN / "hospitalisation.csv")
hos = hos.rename(columns={"orgunitlevel2": "province", "orgunitlevel3": "district",
                          "organisationunitcode": "district_code",
                          "organisationunitname": "facility_type",
                          "Total admissions (including HC admissions)": "admissions",
                          "Average length of stay_Hospitalisation": "avg_los_days"})
hos["province"] = norm_province(hos["province"])
hos["district"] = norm_district(hos["district"])
hos["facility_type"] = hos["facility_type"].str.strip()
hos["avg_los_missing"] = hos["avg_los_days"].isna()   # keep NaN, do NOT fill with 0
hos.to_csv(OUT / "fact_hospitalisation.csv", index=False)
report.append(f"Hospitalisation: {len(hos)} rows; avg_los missing in {int(hos['avg_los_missing'].sum())} "
              f"(kept as NULL, not 0). Types with missing LOS: "
              f"{hos.loc[hos.avg_los_missing,'facility_type'].value_counts().to_dict()}")

# --------------------------------------------------------------- blood bank
bb = pd.read_csv(IN / "blood-bank.csv")
bb.columns = ["province", "district", "facility_type", "bloodtype", "units"]
bb["province"] = norm_province(bb["province"])
bb["district"] = norm_district(bb["district"])
bb["blood_group"] = bb["bloodtype"].str.replace(" Used", "", regex=False).str.strip()
bb["metric"] = "units_used"
bb = bb[["province", "district", "facility_type", "blood_group", "metric", "units"]]
bb.to_csv(OUT / "fact_blood_usage.csv", index=False)
report.append(f"Blood bank: {len(bb)} rows, {bb['units'].sum()} units used; blood groups: "
              f"{sorted(bb['blood_group'].unique())}. Records are ANNUAL-style totals, no dates, no stock.")

# province consistency between sources
off_prov = fac.drop_duplicates("district").set_index("district")["province"]
for name, df in [("hospitalisation", hos), ("blood_bank", bb)]:
    mism = df[df["province"] != df["district"].map(off_prov)][["district", "province"]].drop_duplicates()
    report.append(f"Province check {name}: {len(mism)} district/province mismatches vs official codes"
                  + (f" -> {mism.values.tolist()}" if len(mism) else ""))

# ------------------------------------------------- district master dimension
base = fac.drop_duplicates("district")[["adm2_pcode", "district", "province"]].set_index("district")
hosp_pts = fac[fac["tier"] == "hospital"]
ref = (hosp_pts.groupby("district")[["lat", "lon"]].median()
       .reindex(base.index))
all_pts = fac.groupby("district")[["lat", "lon"]].median()
ref = ref.fillna(all_pts)
ref["ref_point_basis"] = np.where(hosp_pts.groupby("district").size().reindex(base.index).fillna(0) > 0,
                                  "median of OSM hospital-tier points", "median of all OSM points")
osm_counts = fac.pivot_table(index="district", columns="tier", values="id", aggfunc="count",
                             fill_value=0).add_prefix("osm_")
adm = hos.groupby("district").agg(total_admissions=("admissions", "sum"),
                                  district_code=("district_code", "first"))
hosp_adm = (hos[hos.facility_type.str.contains("Hospital")].groupby("district")["admissions"]
            .sum().rename("hospital_admissions"))
blood_tot = bb.groupby("district")["units"].sum().rename("blood_units_used")
mix = (bb.pivot_table(index="district", columns="blood_group", values="units", aggfunc="sum", fill_value=0))
mix = mix.div(mix.sum(axis=1), axis=0).add_prefix("share_")
dim = (base.join(ref).join(osm_counts).join(hf_district).join(adm).join(hosp_adm)
       .join(blood_tot).join(mix))
dim["blood_units_per_1000_hosp_admissions"] = (dim["blood_units_used"] /
                                               dim["hospital_admissions"] * 1000).round(1)
dim = dim.reset_index()
dim.to_csv(OUT / "dim_district.csv", index=False)
report.append(f"dim_district: {len(dim)} rows. Districts with no OSM hospital-tier point: "
              f"{dim.loc[dim.ref_point_basis!='median of OSM hospital-tier points','district'].tolist()}")
report.append(f"Districts with no blood bank rows: {sorted(set(dim.district)-set(bb.district))}")

# ------------------------------------------------------------- distances
R = 6371.0088
lat, lon = np.radians(dim["lat"].values), np.radians(dim["lon"].values)
dlat, dlon = lat[:, None] - lat[None, :], lon[:, None] - lon[None, :]
a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
hav = 2 * R * np.arcsin(np.sqrt(a))
dist = pd.DataFrame(hav.round(1), index=dim["district"], columns=dim["district"])
dist.to_csv(OUT / "district_distance_km.csv")
ROAD_FACTOR, SPEED_KMH = 1.4, 40.0          # ESTIMATES - replace with OSRM routing later
pairs = dist.rename_axis(index="from_district", columns="to_district").stack().rename("straight_km").reset_index()
pairs = pairs[pairs.from_district != pairs.to_district]
pairs["est_road_km"] = (pairs.straight_km * ROAD_FACTOR).round(1)
pairs["est_transit_hours"] = (pairs.est_road_km / SPEED_KMH).round(2)
pairs.to_csv(OUT / "district_pairs.csv", index=False)
report.append(f"Distances: {len(pairs)} district pairs; est. transit = straight_km x {ROAD_FACTOR} / "
              f"{SPEED_KMH} km/h (placeholder, not routed).")

(OUT / "data_quality_report.md").write_text("# Data quality report\n\n" +
                                            "\n".join(f"- {r}" for r in report) + "\n")
print("\n".join(report))
