"""
MediPredict - Step 2: add REAL national blood-system figures (RBC / Blood Transfusion Division,
Annual Report 2024) and compare them with blood-bank.csv.

Source: https://www.rbc.gov.rw/fileadmin/user_upload/report_2024/Services_Department_Blood_Transfusion_Division_ANNUAL_REPORT_2024.pdf
Tables used: 1.5, 1.6, 1.9, 1.10, 3.1, 4.3 (numbers typed from the report; totals are asserted below).
Run:  python 02_add_rbc_2024_reference.py [BLOOD_BANK_CSV] [OUT_DIR]
"""
import sys
from pathlib import Path
import pandas as pd

BB = Path(sys.argv[1] if len(sys.argv) > 1 else "blood-bank.csv")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "processed")
OUT.mkdir(parents=True, exist_ok=True)
SRC = "RBC Blood Transfusion Division Annual Report 2024"

# --- Table 1.9: units collected by blood group, per Regional Centre (RCBT)
GROUPS = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]
t19 = {
    "Kigali":     [13122, 1279, 6457, 411, 4953, 277, 1183, 77],
    "Butare":     [6743, 652, 3286, 214, 2595, 125, 496, 39],
    "Karongi":    [6580, 459, 3137, 156, 2463, 145, 530, 22],
    "Ruhengeri":  [6881, 542, 3748, 158, 2637, 143, 624, 18],
    "Rwamagana":  [6965, 499, 3379, 185, 2489, 141, 560, 13],
}
TOTALS = [40291, 3431, 20007, 1124, 15137, 831, 3393, 169]
RCBT_REGION = {"Kigali": "Kigali City", "Butare": "South", "Karongi": "West",
               "Ruhengeri": "North", "Rwamagana": "East"}   # RCBTs sit in each province + CoK
rows = [(r, RCBT_REGION[r], g, u, 2024, SRC) for r, v in t19.items() for g, u in zip(GROUPS, v)]
bg = pd.DataFrame(rows, columns=["rcbt", "region", "blood_group", "units_collected", "year", "source"])
for g, tot in zip(GROUPS, TOTALS):                       # integrity check against printed totals
    assert bg.loc[bg.blood_group == g, "units_collected"].sum() == tot, g
assert bg["units_collected"].sum() == 84383
bg.to_csv(OUT / "rbc_2024_collected_by_group_rcbt.csv", index=False)

# --- national indicators (units)
ind = [
    ("units_collected", 2024, 84383, "Table 1.6"), ("units_collected", 2023, 86812, "Table 1.6"),
    ("units_planned_collection", 2024, 89148, "Table 1.5"),
    ("hospital_demand_units_rcbt", 2024, 55852, "Table 3.1"),
    ("hospital_supply_units_rcbt", 2024, 55852, "Table 3.1"),
    ("hospital_demand_units_zipline", 2024, 71346, "Table 3.1"),
    ("hospital_supply_units_zipline", 2024, 70985, "Table 3.1"),
    ("hospital_demand_units_total", 2024, 127198, "Table 3.1"),
    ("hospital_supply_units_total", 2024, 126837, "Table 3.1"),
    ("unmet_demand_units_total", 2024, 127198 - 126837, "derived: demand - supply"),
    ("blood_donors", 2024, 58688, "Table 1.8"),
]
ind = pd.DataFrame(ind, columns=["indicator", "year", "value", "table"]); ind["source"] = SRC
ind.to_csv(OUT / "rbc_2024_national_indicators.csv", index=False)

# --- Table 4.3 national: top transfusion indications (components: RBC, platelets, FFP, cryo)
ic = pd.DataFrame([
    ("Anemia", 19973, 1527, 276, 37), ("PPH (postpartum haemorrhage)", 8428, 2336, 2394, 1208),
    ("Bleeding", 3851, 813, 536, 222), ("Thrombocytopenia", 433, 3275, 123, 12),
    ("Sepsis", 1773, 992, 210, 46), ("Cancers", 1769, 918, 97, 30),
    ("Abortion", 1611, 124, 91, 29), ("Malaria", 875, 287, 16, 2),
    ("Uterine rupture", 344, 146, 147, 55)],
    columns=["indication", "rbc", "platelets", "ffp", "cryo"])
ic["total_components"] = ic[["rbc", "platelets", "ffp", "cryo"]].sum(axis=1)
ic["note"] = "top indications only; report's all-indication total is 55,006 components"
ic["source"] = SRC + ", Table 4.3"
ic.to_csv(OUT / "rbc_2024_transfusion_indications.csv", index=False)

# --- compare blood-bank.csv usage mix vs real national collection mix
bb = pd.read_csv(BB)
bb["blood_group"] = bb["bloodtype"].str.replace(" Used", "", regex=False).str.strip()
used = bb.groupby("blood_group")["Total"].sum()
real = pd.Series(dict(zip(GROUPS, TOTALS)))
cmp_ = pd.DataFrame({"blood_bank_csv_units_used": used, "blood_bank_csv_share_pct": (used / used.sum() * 100).round(1),
                     "rbc_collected_2024_units": real, "rbc_collected_share_pct": (real / real.sum() * 100).round(1)})
cmp_["share_ratio_csv_vs_rbc"] = (cmp_["blood_bank_csv_share_pct"] / cmp_["rbc_collected_share_pct"]).round(2)
cmp_.loc[GROUPS].to_csv(OUT / "blood_group_mix_comparison.csv", index_label="blood_group")
print(cmp_.loc[GROUPS].to_string())
neg = ["O-", "A-", "B-", "AB-"]
print("Rh-negative share  blood-bank.csv: %.1f%%   RBC 2024 collected: %.1f%%"
      % (cmp_.loc[neg, "blood_bank_csv_share_pct"].sum(), cmp_.loc[neg, "rbc_collected_share_pct"].sum()))
print("Unmet demand 2024: %d of %d units = %.2f%%" % (127198 - 126837, 127198, (127198 - 126837) / 127198 * 100))
