# Data

**Source:** `freMTPL2freq` and `freMTPL2sev` from the R package
[CASdatasets](https://github.com/dutangc/CASdatasets) (Dutang & Charpentier), French motor third-party
liability portfolio, ~2011-2013. **License:** the package is GPL (>= 2); data are used here for
research/education. Neither dataset is committed to this repo.

**Obtain:** `python -m scripts.prepare_data` downloads both `.rda` files from the CASdatasets GitHub
repo into `data/raw/`. If the download is blocked, fetch them manually from
<https://github.com/dutangc/CASdatasets/tree/master/data> and place them in `data/raw/`.

| Column | Meaning |
|---|---|
| IDpol | Policy id (joins frequency and severity) |
| ClaimNb | Number of claims in the exposure period |
| Exposure | Policy exposure in years |
| VehPower | Vehicle power (ordered integer) |
| VehAge | Vehicle age, years |
| DrivAge | Driver age, years |
| BonusMalus | Bonus-malus (50 = best, 100 = neutral, >100 = malus) |
| VehBrand | Anonymised brand group (B1..B14) |
| VehGas | Diesel / Regular |
| Area | Anonymised density-based area code (A..F) |
| Density | Inhabitants per km2 in the driver's city |
| Region | French region (1970-2015 classification) |
| ClaimAmount (sev) | Individual claim cost, one row per claim |

**Limitations**
- No date column: an out-of-time holdout is not possible, so a stratified random split by policy is used.
- Exposure values above 1 year occur and are data errors; they are capped at 1 (documented in `configs/config.yaml`).
- Some claim amounts are fixed IRSA-IDA convention amounts; claim amounts are not inflation-adjusted.
- No severity-relevant detail (claim type, injury vs property); severity has a heavy tail.
- Claim counts in the frequency file may not match severity rows for every policy; see `data/processed/quality_report.json` after running preparation.
