# Module B — Time-Series Drift Predictor

## Purpose

Module B predicts a component parameter's value at 168 hours using only its 0-hour and 24-hour measurements. It is an early-screening aid: an output of `CONTINUE_SCREENING` means no early drift rejection was found, not that the component is flight-qualified.

## Leakage-safe flow

```text
Long-form or Value_0h/Value_24h input
  -> schema validation and canonical-unit conversion
  -> discard 96 h, 168 h, and hidden target columns
  -> derive early-only features
  -> exact-context model, then parameter fallback if needed
  -> 168 h point forecast plus danger-side uncertainty bound
  -> calculated safety-slope comparison
  -> parameter evidence and component decision
```

Training and evaluation group folds by `lot_id`; measurements from one lot never appear in both sides of a validation fold or train/test split.

## Inputs

The preferred format is the Module A long-form schema:

```text
component_id, lot_id, part_number, parameter, time_h, value, unit,
test_condition_id, temperature_c, voltage_v, test_mode
```

Optional engineering and label fields are:

```text
datasheet_min, datasheet_max, delta_limit,
actual_value_168h, is_anomaly, defect_label, qa_approved,
qa_disposition, defect_type
```

`datasheet_min`, `datasheet_max`, and `delta_limit` are interpreted in the row's source `unit` and converted to the configured canonical unit.

Wide organizer data is also accepted. It must contain the identity/context fields plus:

```text
Value_0h, Value_24h, Value_168h
```

`actual_value_168h` may replace `Value_168h` during training. The aliases `parameter_name`, `temperature_C`, and `bias_voltage` are normalized automatically.

## Forecast models and baselines

For each `part_number x parameter x test_condition_id`, Module B evaluates Ridge regressions at configured regularization strengths and a Huber regression. Selection uses lowest lot-grouped cross-validated MAE. A parameter-level model is fitted as a fallback for a previously unseen part/condition.

Every result includes the required baselines:

```text
Persistence:          x_hat_168 = x_24
Linear extrapolation: x_hat_168 = x_24 + 6(x_24 - x_0)
                                  = 7x_24 - 6x_0
```

Model features are deterministic functions of the two allowed inputs: `value_0h`, `value_24h`, early delta, early slope, relative drift, and log ratio.

## Safety slope

All drift comparisons are direction-aware. Increasing leakage, Iddq, and delay is dangerous; decreasing output-HIGH voltage is dangerous; configured two-sided parameters use absolute drift.

The predicted total drift rate is:

```text
s_pred = direction((x_hat_168 - x_0) / 168)
```

Available safety candidates are:

```text
s_delta = delta_limit / 168
s_hist  = configured upper quantile of known-good 0-to-168 h drift
s_guard = rate that would consume the remaining datasheet guard-band headroom
```

The binding threshold is deliberately conservative:

```text
s_safe = min(s_delta, s_hist, s_guard)  # among available candidates
```

The point forecast triggers `EARLY_REJECT` when `s_pred > s_safe`. A second sentinel protects against nonlinear acceleration that a robust fit may smooth over: it triggers only when the calibrated danger-side prediction interval crosses `s_safe` and the mandatory linear baseline exceeds `linear_sentinel_multiplier x s_safe`.

## Decisions

```text
CONTINUE_SCREENING  no Module B early-reject condition
EARLY_REJECT        unsafe forecast drift, guard crossing, or forecast spec failure
STATIC_FAIL         0 h or 24 h value already violates a datasheet limit
RETEST_REQUIRED     missing/invalid input, conflicting engineering data, or no model
```

Static failure has highest precedence. Component summaries retain the worst parameter decision and all reason codes.

## Explainability

Each parameter result exposes:

- 0 h and 24 h values in canonical units;
- point and conservative 168 h forecasts;
- persistence and linear baseline forecasts;
- predicted, conservative, and linear danger-directed slopes;
- every safety-slope candidate and which one bound the decision;
- selected model and exact-context/fallback scope;
- cross-validated model and baseline MAE;
- standardized linear coefficients and per-feature contributions;
- decision, risk score, and stable reason codes.

For Ridge and Huber models, the intercept plus listed contributions and any physical-domain clipping adjustment reconstructs the reported forecast.

## Commands

```bash
.venv/bin/ess-module-b fit \
  --input data/synthetic.csv \
  --output artifacts/module_b_predictor.joblib

.venv/bin/ess-module-b forecast \
  --input data/one_lot.csv \
  --artifact artifacts/module_b_predictor.joblib \
  --output artifacts/module_b_forecast.json

.venv/bin/ess-module-b evaluate \
  --input data/synthetic.csv \
  --output artifacts/module_b_acceptance_evaluation.json
```

The Python API exposes `fit_predictor`, `forecast_lot`, and `ModuleBEngine`. The HTTP API exposes `GET /health`, `GET /v1/module-b/model-info`, and `POST /v1/module-b/forecast-lot`.

## Qualification boundary

The checked-in model and report validate software behavior on deterministic synthetic data. Before operational use, retrain on representative approved lots, replace generic guard bands with authorized engineering limits, lock configuration/model versions, and validate false-negative and false-reject rates under the applicable QA procedure.
