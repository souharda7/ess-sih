# ESS Module A — Dynamic Outlier Detection

An offline, explainable screening layer for burn-in/ESS parametric data. It preserves static specification checks and adds lot-relative, historical, drift, Isolation Forest, and robust multivariate evidence.

To run the app, [Click here](https://ess-inspector.streamlit.app/)

## What it returns

The scorer accepts a complete lot in long form and returns evidence for every parameter/checkpoint plus one status per component:

```text
NORMAL | MONITOR | QUARANTINE | STATIC_FAIL | RETEST_REQUIRED
```

`QUARANTINE` means engineering review or retest is recommended. It is not an autonomous hardware disposition.

## Architecture

```text
CSV/JSON lot
  → validation and unit normalization
  → current-lot + historical reference features
  → robust Z / IQR / percentile / slope rules
  → Isolation Forest + robust Mahalanobis distance
  → explainable evidence fusion
  → parameter results, component summaries, and lot alerts
```

The robust Z-score is:

```text
z_robust = (value - median) / max(1.4826 × MAD, epsilon)
```

It is calculated independently against the current lot and approved historical references. Slope values receive the same treatment.

## Install

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e '.[dev]'
```

## Run the demo

```bash
.venv/bin/python examples/demo.py
```

The demo fits approved synthetic reference lots and scores an unseen lot at 24 hours.

## Interactive QA Dashboard

The project includes a full-featured, professional QA dashboard built with Streamlit. It connects directly to the ESS engine to provide real-time multivariate anomaly scoring (Isolation Forest & Mahalanobis), drift trajectory analysis, and component-level deep-dives.

To run it locally:
```bash
streamlit run app.py
```

## Deployment

The project is fully configured for easy cloud deployment:

* **Streamlit Community Cloud (Frontend):** A flat `requirements.txt` is included at the root. Simply connect the repository to Streamlit Cloud, and it will deploy the interactive dashboard instantly.
* **Render (Infrastructure as Code):** A `render.yaml` file is provided to automatically deploy the FastAPI backend (`ess-backend`) and the Streamlit frontend (`ess-frontend`) as separate web services.

## Command-line workflow

Generate PS-shaped development data (includes up to 5% random defect fluctuations per normal lot, plus specific lot-wide shifts):

```bash
.venv/bin/ess-module-a generate \
  --output data/synthetic.csv \
  --lots 30 \
  --components 100
```

Fit and version the historical reference:

```bash
.venv/bin/ess-module-a fit \
  --input data/synthetic.csv \
  --output artifacts/reference.json
```

Score a CSV containing exactly one lot:

```bash
.venv/bin/ess-module-a score \
  --input data/one_lot.csv \
  --reference artifacts/reference.json \
  --as-of 24 \
  --output artifacts/score.json
```

Run lot-safe evaluation:

```bash
.venv/bin/ess-module-a evaluate \
  --input data/synthetic.csv \
  --as-of 168 \
  --output artifacts/evaluation.json
```

## API

Start the local service:

```bash
.venv/bin/ess-module-a serve \
  --reference artifacts/reference.json
```

Endpoints:

```text
GET  /health
GET  /v1/module-a/model-info
POST /v1/module-a/score-lot
```

The score request is:

```json
{
  "as_of_h": 24,
  "measurements": [
    {
      "component_id": "C001",
      "lot_id": "LOT_TEST",
      "part_number": "PN_LOGIC_A",
      "parameter": "leakage_current",
      "time_h": 0,
      "value": 10.0,
      "unit": "uA",
      "test_condition_id": "PN_LOGIC_A_125C_NOMINAL",
      "temperature_c": 125,
      "voltage_v": 3.3,
      "test_mode": "static_bias"
    }
  ]
}
```

Measurements later than `as_of_h` are explicitly ignored and reported as `FUTURE_MEASUREMENT_IGNORED`; they never enter early features.

## Input columns

Required:

```text
component_id, lot_id, part_number, parameter, time_h,
value, unit, test_condition_id
```

Required when configured for a parameter:

```text
temperature_c, voltage_v, test_mode
```

Recommended context:

```text
tester_id, chamber_id, socket_id
```

Engineering limits, units, transformations, direction of danger, and minimum peer counts live in [`configs/parameters.yaml`](configs/parameters.yaml).

## Evidence fusion

Individual warnings remain in `reason_codes`. Because one component is checked across many parameters and checkpoints, a QA status requires corroborating evidence to control repeated-test false alarms:

- A valid static violation produces `STATIC_FAIL`.
- Invalid or incomplete required data produces `RETEST_REQUIRED`.
- An extreme robust Z-score produces `QUARANTINE` by itself.
- Multiple independent severe sources produce `QUARANTINE`.
- Multiple corroborating severe/warning sources produce `MONITOR`.
- Uncorroborated weak evidence is retained for audit without automatically changing the component status.

The initial thresholds are high-recall defaults and are configuration-controlled. Replace them with ISRO-approved limits and validation results when official data arrives.

## External data adapters

`ess_module_a.adapters.adapt_wide_checkpoints` converts organizer-style `Value_0h`/`Value_24h` columns to long form.

`adapt_nasa_igbt_leakage` converts the public NASA IGBT archive's two-column `LeakageIV.csv` curves into comparable snapshot rows at a selected voltage. NASA's dataset contains too few aged devices to establish lot thresholds, so it is intentionally used only to validate ingestion and measurement handling. [NASA dataset](https://data.nasa.gov/dataset/insulated-gate-bipolar-transistor-igbt-accelerated-aging)

## Verification

```bash
.venv/bin/pytest
.venv/bin/pytest --cov=ess_module_a --cov-report=term-missing
```

The suite covers the 45 µA-in-a-10 µA-lot scenario, direction-aware thresholds, unit conversion, missing/duplicate readings, future-data leakage, static precedence, whole-lot alerts, adapters, lot-safe splits, and API behavior.

The checked-in acceptance report at [`artifacts/acceptance_evaluation.json`](artifacts/acceptance_evaluation.json) was produced from 30 synthetic lots × 100 components with whole-lot splitting. On its unseen test partition it recorded 12/12 detected injected defects, zero false negatives, and a 2.04% healthy-component flag rate. These numbers validate the software scenarios only; they are not estimates of real hardware performance.

## Current boundaries

- Synthetic results demonstrate implementation behavior, not real flight qualification.
- Reference profiles must be built from training or QA-approved lots and are never updated during scoring.
- Final thresholds, peer definitions, limits, and acceptable false-reject rate require the official ISRO data and engineering procedure.
- Module B's 168-hour regression forecast is intentionally outside this package.
