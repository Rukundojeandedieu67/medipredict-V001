# Data quality report

- OSM features: 1372 raw -> 1310 after node/polygon de-duplication (62 removed).
- OSM tier counts: {'pharmacy': 461, 'health_center': 341, 'other': 143, 'health_post': 141, 'hospital': 115, 'clinic': 109}
- Booklet: 2226 rows, districts after normalisation: 30, unmatched to official list: []
- Hospitalisation: 153 rows; avg_los missing in 33 (kept as NULL, not 0). Types with missing LOS: {'Health Post': 15, 'Health Post 2nd Generation': 7, 'Prison Clinic': 6, 'Medical Clinic': 2, 'Medicalized health Center': 1, 'Specialised Hospital': 1, 'Private Hospital': 1}
- Blood bank: 400 rows, 43633 units used; blood groups: ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']. Records are ANNUAL-style totals, no dates, no stock.
- Province check hospitalisation: 0 district/province mismatches vs official codes
- Province check blood_bank: 0 district/province mismatches vs official codes
- dim_district: 30 rows. Districts with no OSM hospital-tier point: []
- Districts with no blood bank rows: []
- Distances: 870 district pairs; est. transit = straight_km x 1.4 / 40.0 km/h (placeholder, not routed).
