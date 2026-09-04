"""
ESS QA Inspector Dashboard  ·  Module A + Module B integration
===============================================================
Uses the production Module A and Module B engines for dynamic outlier screening
and explainable 168-hour drift forecasting. The dashboard only visualises engine
results; classification and early-rejection decisions stay in the domain layer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the installed package importable from source when running via
# `streamlit run app.py` from the project root.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ess_module_a.config import load_config
from ess_module_a.engine import ModuleAEngine
from ess_module_b.config import load_config as load_module_b_config
from ess_module_b.engine import ModuleBEngine

# ─────────────────────────────────────────────────────────────────────────────
# Constants / paths
# ─────────────────────────────────────────────────────────────────────────────
REFERENCE_PATH = "artifacts/reference.json"
CONFIG_PATH    = "configs/parameters.yaml"
DATA_PATH      = "data/synthetic.csv"
MODULE_B_ARTIFACT_PATH = "artifacts/module_b_predictor.joblib"
MODULE_B_CONFIG_PATH = "configs/module_b.yaml"
MODULE_B_REPORT_PATH = "artifacts/module_b_acceptance_evaluation.json"

MODULE_A_VIEW = "Module A · Dynamic Outlier Detection"
MODULE_B_VIEW = "Module B · 168 h Drift Predictor"

STATUS_COLOR = {
    "NORMAL":          "#2ECC71",
    "MONITOR":         "#F39C12",
    "QUARANTINE":      "#E74C3C",
    "STATIC_FAIL":     "#8E44AD",
    "RETEST_REQUIRED": "#95A5A6",
}

STATUS_PRIORITY = {
    "NORMAL": 0, "MONITOR": 1, "QUARANTINE": 2,
    "RETEST_REQUIRED": 3, "STATIC_FAIL": 4,
}

DRIFT_COLOR = {
    "CONTINUE_SCREENING": "#2ECC71",
    "EARLY_REJECT": "#E74C3C",
    "STATIC_FAIL": "#8E44AD",
    "RETEST_REQUIRED": "#95A5A6",
}

DRIFT_PRIORITY = {
    "CONTINUE_SCREENING": 0,
    "EARLY_REJECT": 1,
    "RETEST_REQUIRED": 2,
    "STATIC_FAIL": 3,
}

SPEC_DISPLAY = {
    "leakage_current":    "≤ 50 µA",
    "iddq":               "≤ 80 µA",
    "propagation_delay":  "≤ 20 ns",
    "output_high_voltage":"≥ 2.4 V",
    "threshold_voltage":  "0.35 – 0.95 V",
}

SPEC_LIMITS: dict[str, tuple] = {
    "leakage_current":    ("max", 50.0),
    "iddq":               ("max", 80.0),
    "propagation_delay":  ("max", 20.0),
    "output_high_voltage":("min",  2.4),
    "threshold_voltage":  ("two",  0.35, 0.95),
}

# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Space-Grade ESS QA Dashboard",
    
    layout="wide",
)
st.title("Electronic Stress Screening (ESS) — QA Inspector")


# ─────────────────────────────────────────────────────────────────────────────
# Engine + data loading  (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading reference model…")
def load_engine() -> ModuleAEngine:
    return ModuleAEngine.load(REFERENCE_PATH, load_config(CONFIG_PATH))


@st.cache_resource(show_spinner="Loading Module B drift predictor…")
def load_module_b_engine() -> ModuleBEngine:
    return ModuleBEngine.load(
        MODULE_B_ARTIFACT_PATH,
        load_module_b_config(MODULE_B_CONFIG_PATH),
    )


@st.cache_data(show_spinner="Loading measurement data…")
def load_measurements() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner="Loading Module B acceptance evidence…")
def load_module_b_report() -> dict:
    report_path = Path(MODULE_B_REPORT_PATH)
    if not report_path.exists():
        return {}
    with report_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner="Scoring selected lot (takes ~5 seconds)…")
def score_single_lot(_engine_id: int, lot_id: str) -> dict:
    """Score a single lot lazily to prevent 5-minute startup times."""
    engine = load_engine()
    df     = load_measurements()
    lot_df = df[df["lot_id"] == lot_id].copy()
    try:
        return engine.score_lot(lot_df, as_of_h=168.0)
    except Exception as exc:
        return {"error": str(exc)}


@st.cache_data(show_spinner="Forecasting the selected lot from 0 h and 24 h…")
def forecast_single_lot(_engine_id: int, lot_id: str) -> dict:
    """Run Module B with early checkpoints only; never expose 168 h to inference."""

    engine = load_module_b_engine()
    df = load_measurements()
    early_df = df[
        (df["lot_id"] == lot_id)
        & (df["time_h"].isin([engine.config.baseline_h, engine.config.early_h]))
    ].copy()
    try:
        return engine.forecast_lot(early_df)
    except Exception as exc:
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: flatten a lot's engine result into tidy DataFrames
# ─────────────────────────────────────────────────────────────────────────────

def lot_to_frames(result: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Returns
    -------
    param_df      one row per (component, parameter, time_h)
    comp_df       one row per component  (worst-status roll-up)
    lot_alerts    list of lot-level alert dicts
    """
    param_df   = pd.DataFrame(result.get("parameter_results", []))
    comp_df    = pd.DataFrame(result.get("component_results",  []))
    lot_alerts = result.get("lot_alerts", [])
    return param_df, comp_df, lot_alerts


def _first_numeric(frame: pd.DataFrame, column: str) -> float | None:
    """Return the first finite numeric value from a result column."""

    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else None


def _display_number(value: object, unit: str = "") -> str:
    """Format optional engine values safely for inspector metrics."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    if pd.isna(number):
        return "Unavailable"
    suffix = f" {unit}" if unit else ""
    return f"{number:.5g}{suffix}"


def render_module_b(raw_df: pd.DataFrame) -> None:
    """Render the explainable Module B early-rejection workflow."""

    st.markdown("## Module B — Time-Series Drift Predictor")
    st.caption(
        "Forecasts each parameter at 168 h using only its 0 h and 24 h readings. "
        "The trained model, uncertainty bound, static limits, and calculated safety "
        "slope jointly determine whether a component should be rejected early."
    )

    try:
        drift_engine = load_module_b_engine()
    except Exception as exc:
        st.error(f"Unable to load the Module B predictor: {exc}")
        st.info(
            f"Expected artifact: `{MODULE_B_ARTIFACT_PATH}` · "
            f"configuration: `{MODULE_B_CONFIG_PATH}`"
        )
        return

    available_lots = sorted(str(value) for value in raw_df["lot_id"].dropna().unique())
    st.sidebar.header("Module B Filters")
    selected_lot = st.sidebar.selectbox(
        "Lot ID",
        available_lots,
        key="module_b_lot",
    )
    lot_parameters = sorted(
        str(value)
        for value in raw_df.loc[raw_df["lot_id"] == selected_lot, "parameter"]
        .dropna()
        .unique()
    )
    selected_param = st.sidebar.selectbox(
        "Parameter",
        lot_parameters,
        key="module_b_parameter",
    )

    with st.sidebar.expander("Module B Model Configuration", expanded=False):
        st.json(drift_engine.model_info())

    lot_result = forecast_single_lot(id(drift_engine), selected_lot)
    if "error" in lot_result:
        st.error(f"Module B forecast failed for {selected_lot}: {lot_result['error']}")
        return

    prediction_df = pd.DataFrame(lot_result.get("prediction_results", []))
    component_df = pd.DataFrame(lot_result.get("component_results", []))
    if prediction_df.empty or component_df.empty:
        st.warning("Module B returned no forecastable component series for this lot.")
        return

    st.markdown(f"### Early-Screening Overview · **{selected_lot}**")
    decision_counts = component_df["decision"].value_counts()
    total = len(component_df)
    n_continue = int(decision_counts.get("CONTINUE_SCREENING", 0))
    n_reject = int(decision_counts.get("EARLY_REJECT", 0))
    n_static = int(decision_counts.get("STATIC_FAIL", 0))
    n_retest = int(decision_counts.get("RETEST_REQUIRED", 0))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Components", total)
    c2.metric("Continue Screening", n_continue)
    c3.metric("Early Reject", n_reject, delta_color="inverse")
    c4.metric("Static Fail", n_static, delta_color="inverse")
    c5.metric("Retest Required", n_retest, delta_color="inverse")

    action_count = n_reject + n_static + n_retest
    if action_count:
        st.error(
            f"{action_count} of {total} components require QA action at 24 h: "
            f"{n_reject} early rejection, {n_static} static failure, and "
            f"{n_retest} retest."
        )
    else:
        st.success(
            "No component is currently flagged for early rejection or retest. "
            "Continue screening; this is not final flight acceptance."
        )

    st.caption(
        f"Inference checkpoint: {lot_result.get('input_as_of_h', 24):g} h · "
        f"Forecast target: {lot_result.get('target_h', 168):g} h · "
        "later measurements were not supplied to the predictor."
    )

    acceptance_report = load_module_b_report()
    held_out = acceptance_report.get("test", {})
    held_out_screening = held_out.get("screening", {})
    held_out_regression = held_out.get("regression", {})
    if held_out_screening and held_out_regression:
        with st.expander("Held-Out Acceptance Evidence", expanded=False):
            recall = float(held_out_screening.get("defect_recall", 0.0))
            tp = int(held_out_screening.get("true_positives", 0))
            fn = int(held_out_screening.get("false_negatives", 0))
            false_flag_rate = float(held_out_screening.get("false_flag_rate", 0.0))
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Defective-Part Recall", f"{recall:.2%} ({tp}/{tp + fn})")
            e2.metric("False Negatives", fn, delta_color="inverse")
            e3.metric("Aggregate MAE", f"{held_out_regression.get('mae', 0.0):.3f}")
            e4.metric("Healthy-Part Flag Rate", f"{false_flag_rate:.2%}")
            st.caption(
                "These values come from the checked-in lot-held-out synthetic acceptance "
                "report; they are not calculated from the currently selected lot and are "
                "not hardware-qualification claims."
            )

    validation_issues = lot_result.get("validation_issues", [])
    if validation_issues:
        with st.expander(
            f"Input Validation Issues ({len(validation_issues)})",
            expanded=False,
        ):
            st.dataframe(pd.DataFrame(validation_issues), use_container_width=True)

    st.divider()
    st.markdown("### Component Decisions and Forecast Risk")
    col_decision, col_risk = st.columns(2)

    with col_decision:
        counts = component_df["decision"].value_counts().reset_index()
        counts.columns = ["Decision", "Count"]
        fig_decision = px.pie(
            counts,
            names="Decision",
            values="Count",
            color="Decision",
            color_discrete_map=DRIFT_COLOR,
            title="24 h Early-Screening Decisions",
            hole=0.42,
        )
        fig_decision.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_decision, use_container_width=True)

    with col_risk:
        fig_risk = px.histogram(
            component_df,
            x="risk_score",
            color="decision",
            color_discrete_map=DRIFT_COLOR,
            nbins=20,
            title="Component Forecast-Risk Distribution",
            labels={"risk_score": "Risk Score (0–1)", "decision": "Decision"},
        )
        fig_risk.update_layout(bargap=0.05)
        st.plotly_chart(fig_risk, use_container_width=True)

    st.divider()
    st.markdown(f"### 168 h Forecast Evidence — **{selected_param}**")
    parameter_df = prediction_df[prediction_df["parameter"] == selected_param].copy()
    parameter_df["decision_priority"] = parameter_df["decision"].map(DRIFT_PRIORITY).fillna(0)
    parameter_df = parameter_df.sort_values(
        ["decision_priority", "risk_score"],
        ascending=[False, False],
    )

    forecast_col, slope_col = st.columns(2)
    with forecast_col:
        hover_fields = {
            "component_id": True,
            "value_0h": ":.5g",
            "value_24h": ":.5g",
            "predicted_value_168h": ":.5g",
            "risk_score": ":.3f",
            "model": True,
            "decision": True,
        }
        fig_forecast = px.scatter(
            parameter_df,
            x="component_id",
            y="predicted_value_168h",
            color="decision",
            color_discrete_map=DRIFT_COLOR,
            hover_data=hover_fields,
            title=f"Predicted 168 h Endpoint — {selected_param}",
            labels={
                "component_id": "Component",
                "predicted_value_168h": f"Predicted Value ({parameter_df['unit'].iloc[0]})",
                "decision": "Decision",
            },
        )
        for value, label, dash, color in [
            (_first_numeric(parameter_df, "datasheet_min"), "Spec minimum", "dash", "#8E44AD"),
            (_first_numeric(parameter_df, "datasheet_max"), "Spec maximum", "dash", "#8E44AD"),
            (_first_numeric(parameter_df, "guard_min"), "Guard minimum", "dot", "#F39C12"),
            (_first_numeric(parameter_df, "guard_max"), "Guard maximum", "dot", "#F39C12"),
        ]:
            if value is not None:
                fig_forecast.add_hline(
                    y=value,
                    line_dash=dash,
                    line_color=color,
                    annotation_text=label,
                )
        fig_forecast.update_layout(xaxis_showticklabels=False)
        st.plotly_chart(fig_forecast, use_container_width=True)

    with slope_col:
        slope_plot = parameter_df.dropna(
            subset=["safety_slope_per_h", "danger_directed_drift_rate_per_h"]
        ).copy()
        fig_slope = px.scatter(
            slope_plot,
            x="safety_slope_per_h",
            y="danger_directed_drift_rate_per_h",
            color="decision",
            size="risk_score",
            color_discrete_map=DRIFT_COLOR,
            hover_name="component_id",
            hover_data={
                "predicted_value_168h": ":.5g",
                "safety_slope_per_h": ":.5g",
                "danger_directed_drift_rate_per_h": ":.5g",
                "risk_score": ":.3f",
            },
            title="Predicted Danger Slope vs. Safety Slope",
            labels={
                "safety_slope_per_h": "Calculated Safety Slope",
                "danger_directed_drift_rate_per_h": "Predicted Danger-Directed Slope",
                "decision": "Decision",
            },
        )
        if not slope_plot.empty:
            axis_min = min(
                0.0,
                float(slope_plot["danger_directed_drift_rate_per_h"].min()),
            )
            axis_max = max(
                float(slope_plot["safety_slope_per_h"].max()),
                float(slope_plot["danger_directed_drift_rate_per_h"].max()),
                1e-9,
            )
            fig_slope.add_trace(
                go.Scatter(
                    x=[axis_min, axis_max],
                    y=[axis_min, axis_max],
                    mode="lines",
                    name="Reject boundary",
                    line=dict(color="#E74C3C", dash="dash"),
                    hoverinfo="skip",
                )
            )
        st.plotly_chart(fig_slope, use_container_width=True)
        st.caption(
            "Points above the diagonal exceed the binding safety slope. Interval, "
            "guard-band, and static-limit rules can also trigger action."
        )

    table_columns = [
        "component_id",
        "value_0h",
        "value_24h",
        "predicted_value_168h",
        "persistence_prediction_168h",
        "linear_extrapolation_prediction_168h",
        "danger_directed_drift_rate_per_h",
        "safety_slope_per_h",
        "model",
        "decision",
        "risk_score",
        "reason_codes",
    ]
    st.dataframe(
        parameter_df[[column for column in table_columns if column in parameter_df.columns]],
        use_container_width=True,
        hide_index=True,
        height=360,
    )
    st.download_button(
        "Download Selected-Parameter Forecasts",
        parameter_df.drop(columns=["decision_priority"], errors="ignore").to_csv(index=False),
        file_name=f"{selected_lot}_{selected_param}_module_b_forecasts.csv",
        mime="text/csv",
    )

    st.divider()
    st.markdown("### Component Forecast Audit")
    action_components = component_df[
        component_df["decision"] != "CONTINUE_SCREENING"
    ]["component_id"].tolist()
    audit_scope = st.radio(
        "Audit scope",
        ["Action required", "All components"],
        horizontal=True,
        key="module_b_audit_scope",
    )
    audit_pool = action_components if audit_scope == "Action required" else component_df["component_id"].tolist()
    if not audit_pool:
        st.success("No components currently require early action in this lot.")
        audit_pool = component_df["component_id"].tolist()

    selected_component = st.selectbox(
        "Component ID",
        sorted(audit_pool),
        key="module_b_component",
    )
    component_predictions = prediction_df[
        prediction_df["component_id"] == selected_component
    ].copy()
    component_summary = component_df[
        component_df["component_id"] == selected_component
    ].iloc[0]
    component_parameters = sorted(component_predictions["parameter"].unique())
    audit_parameter = st.selectbox(
        "Forecast parameter",
        component_parameters,
        index=component_parameters.index(component_summary["highest_risk_parameter"])
        if component_summary["highest_risk_parameter"] in component_parameters
        else 0,
        key="module_b_audit_parameter",
    )
    audit_row = component_predictions[
        component_predictions["parameter"] == audit_parameter
    ].iloc[0]

    decision = str(audit_row["decision"])
    badge_color = DRIFT_COLOR.get(decision, "#777777")
    st.markdown(
        f"**Component `{selected_component}` · `{audit_parameter}`** — "
        f"<span style='background:{badge_color};color:white;padding:3px 10px;"
        f"border-radius:4px;font-weight:bold'>{decision}</span>",
        unsafe_allow_html=True,
    )

    a1, a2, a3, a4, a5, a6 = st.columns(6)
    unit = str(audit_row.get("unit") or "")
    a1.metric("Value 0 h", _display_number(audit_row.get("value_0h"), unit))
    a2.metric("Value 24 h", _display_number(audit_row.get("value_24h"), unit))
    a3.metric(
        "Forecast 168 h",
        _display_number(audit_row.get("predicted_value_168h"), unit),
    )
    a4.metric(
        "Danger Slope",
        _display_number(audit_row.get("danger_directed_drift_rate_per_h"), f"{unit}/h"),
    )
    safety_value = audit_row.get("safety_slope_per_h")
    a5.metric(
        "Safety Slope",
        _display_number(safety_value, f"{unit}/h"),
    )
    risk_value = audit_row.get("risk_score")
    a6.metric(
        "Risk",
        "Unavailable" if pd.isna(risk_value) else f"{float(risk_value):.1%}",
    )

    trajectory_col, evidence_col = st.columns([1.25, 1])
    with trajectory_col:
        fig_trajectory = go.Figure()
        fig_trajectory.add_trace(
            go.Scatter(
                x=[0, 24, 168],
                y=[
                    audit_row["value_0h"],
                    audit_row["value_24h"],
                    audit_row["predicted_value_168h"],
                ],
                mode="lines+markers",
                name="Model forecast",
                line=dict(color=badge_color, width=3),
            )
        )
        fig_trajectory.add_trace(
            go.Scatter(
                x=[168],
                y=[audit_row["persistence_prediction_168h"]],
                mode="markers",
                name="Persistence baseline",
                marker=dict(symbol="diamond", size=11, color="#3498DB"),
            )
        )
        fig_trajectory.add_trace(
            go.Scatter(
                x=[168],
                y=[audit_row["linear_extrapolation_prediction_168h"]],
                mode="markers",
                name="Linear baseline",
                marker=dict(symbol="x", size=12, color="#F39C12"),
            )
        )
        for value, label, dash, color in [
            (audit_row.get("datasheet_min"), "Spec minimum", "dash", "#8E44AD"),
            (audit_row.get("datasheet_max"), "Spec maximum", "dash", "#8E44AD"),
            (audit_row.get("guard_min"), "Guard minimum", "dot", "#F39C12"),
            (audit_row.get("guard_max"), "Guard maximum", "dot", "#F39C12"),
        ]:
            if value is not None and pd.notna(value):
                fig_trajectory.add_hline(
                    y=float(value),
                    line_dash=dash,
                    line_color=color,
                    annotation_text=label,
                )
        fig_trajectory.update_layout(
            title=f"0 h + 24 h → Predicted 168 h ({unit})",
            xaxis_title="Burn-In Time (h)",
            yaxis_title=f"{audit_parameter} ({unit})",
        )
        st.plotly_chart(fig_trajectory, use_container_width=True)

    with evidence_col:
        explanation = audit_row.get("explanation") or {}
        st.markdown("#### Inspector Explanation")
        st.info(explanation.get("summary", "No model explanation is available."))
        reason_codes = audit_row.get("reason_codes") or []
        if reason_codes:
            st.markdown("**Reason codes**")
            st.markdown("  ".join(f"`{code}`" for code in reason_codes))
        else:
            st.markdown("**Reason codes:** `NONE`")

        safety_sources = audit_row.get("safety_slope_sources") or []
        if safety_sources:
            st.markdown("**Safety-slope calculation**")
            source_frame = pd.DataFrame(safety_sources)
            source_frame["binding"] = source_frame["binding"].map(
                {True: "YES", False: ""}
            )
            st.dataframe(source_frame, use_container_width=True, hide_index=True)

    contribution_data = (audit_row.get("explanation") or {}).get("contributions")
    if contribution_data and contribution_data.get("terms"):
        st.markdown("#### Model Feature Contributions")
        contribution_frame = pd.DataFrame(contribution_data["terms"])
        contribution_frame["direction"] = contribution_frame["contribution"].map(
            lambda value: "Raises forecast" if value >= 0 else "Lowers forecast"
        )
        fig_contribution = px.bar(
            contribution_frame,
            x="contribution",
            y="feature",
            orientation="h",
            color="direction",
            color_discrete_map={
                "Raises forecast": "#E74C3C",
                "Lowers forecast": "#3498DB",
            },
            hover_data={
                "value": ":.5g",
                "coefficient": ":.5g",
                "standardized_value": ":.5g",
            },
            title="How each 0 h/24 h-derived feature moves the 168 h forecast",
        )
        st.plotly_chart(fig_contribution, use_container_width=True)

    calibration = (audit_row.get("explanation") or {}).get("calibration") or {}
    baseline_frame = pd.DataFrame(
        [
            {"Method": "Selected model", "168 h estimate": audit_row["predicted_value_168h"]},
            {"Method": "Persistence (Value_24h)", "168 h estimate": audit_row["persistence_prediction_168h"]},
            {"Method": "Linear extrapolation", "168 h estimate": audit_row["linear_extrapolation_prediction_168h"]},
            {"Method": "Conservative endpoint", "168 h estimate": audit_row["conservative_value_168h"]},
        ]
    )
    baseline_col, calibration_col = st.columns(2)
    with baseline_col:
        st.markdown("#### Required Baseline Comparison")
        st.dataframe(baseline_frame, use_container_width=True, hide_index=True)
    with calibration_col:
        st.markdown("#### Model Calibration Evidence")
        if calibration:
            st.json(calibration)
        else:
            st.info("Calibration evidence is unavailable for this series.")

    st.divider()
    st.markdown("### Full Module B Component Summary")
    summary_columns = [
        "component_id",
        "decision",
        "risk_score",
        "highest_risk_parameter",
        "predicted_value_168h",
        "flagged_for_early_rejection",
        "reason_codes",
    ]
    module_b_summary = component_df[summary_columns].copy()
    module_b_summary["decision_priority"] = module_b_summary["decision"].map(DRIFT_PRIORITY)
    module_b_summary = module_b_summary.sort_values(
        ["decision_priority", "risk_score"],
        ascending=[False, False],
    ).drop(columns="decision_priority")
    st.dataframe(module_b_summary, use_container_width=True, hide_index=True, height=420)
    st.caption(
        "Module B · explainable 168 h regression from 0 h and 24 h readings · "
        "CONTINUE_SCREENING means no early-rejection trigger was found; it is not final acceptance."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Module navigation
# ─────────────────────────────────────────────────────────────────────────────
raw_df = load_measurements()
selected_module = st.sidebar.radio(
    "Screening Module",
    [MODULE_A_VIEW, MODULE_B_VIEW],
    key="screening_module",
)

if selected_module == MODULE_B_VIEW:
    render_module_b(raw_df)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Module A sidebar
# ─────────────────────────────────────────────────────────────────────────────
engine = load_engine()
st.sidebar.header("Filters")
available_lots = sorted(raw_df["lot_id"].unique())
selected_lot   = st.sidebar.selectbox("Lot ID", available_lots)

lot_result = score_single_lot(id(engine), selected_lot)
if "error" in lot_result:
    st.error(f"Engine error for {selected_lot}: {lot_result['error']}")
    st.stop()

param_df, comp_df, lot_alerts = lot_to_frames(lot_result)

available_params = sorted(param_df["parameter"].unique()) if not param_df.empty else []
selected_param   = st.sidebar.selectbox("Parameter", available_params)

# Model info in sidebar
with st.sidebar.expander("Model Configuration", expanded=False):
    info = engine.model_info()
    st.json(info)

# ─────────────────────────────────────────────────────────────────────────────
# ① KPI banner
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"### Lot Overview · **{selected_lot}**")

total         = len(comp_df)
n_static_fail = (comp_df["status"] == "STATIC_FAIL").sum()
n_quarantine  = (comp_df["status"] == "QUARANTINE").sum()
n_monitor     = (comp_df["status"] == "MONITOR").sum()
n_normal      = (comp_df["status"] == "NORMAL").sum()
n_retest      = (comp_df["status"] == "RETEST_REQUIRED").sum()
n_flagged     = n_static_fail + n_quarantine   # anything needing action

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Components",        total)
c2.metric("Normal",       n_normal)
c3.metric("Monitor",      n_monitor,     delta_color="inverse")
c4.metric("Quarantine",   n_quarantine,  delta_color="inverse")
c5.metric("Static Fail",  n_static_fail, delta_color="inverse")
c6.metric("Retest Required",       n_retest,      delta_color="inverse")

# ─────────────────────────────────────────────────────────────────────────────
# ② Lot-level alerts banner
# ─────────────────────────────────────────────────────────────────────────────
if lot_alerts:
    st.divider()
    st.markdown("### Lot-Level Alerts")
    for alert in lot_alerts:
        severity = alert.get("severity", "WARNING")
        icon = ""
        msg_parts = [
            f"**{alert['type']}**",
            f"param=`{alert.get('parameter','?')}`",
            f"at `{alert.get('time_h','?')}h`",
            f"Z={alert.get('direction_aware_robust_z', 0.0):.2f}",
        ]
        if "tester_id" in alert:
            msg_parts.append(f"tester=`{alert['tester_id']}`")
        if "chamber_id" in alert:
            msg_parts.append(f"chamber=`{alert['chamber_id']}`")
        msg_parts.append(f"severity=**{severity}**")
        if severity == "SEVERE":
            st.error("  ·  ".join(msg_parts))
        else:
            st.warning("  ·  ".join(msg_parts))

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ③ Component status pie + risk score distribution
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Component Status Overview")
col_pie, col_risk = st.columns(2)

with col_pie:
    status_counts = comp_df["status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    fig_pie = px.pie(
        status_counts,
        names="Status",
        values="Count",
        color="Status",
        color_discrete_map=STATUS_COLOR,
        title="Component Status Distribution",
        hole=0.4,
    )
    fig_pie.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

with col_risk:
    fig_risk = px.histogram(
        comp_df,
        x="risk_score",
        color="status",
        color_discrete_map=STATUS_COLOR,
        nbins=20,
        title="Risk Score Distribution",
        labels={"risk_score": "Risk Score (0–1)", "status": "Status"},
    )
    fig_risk.update_layout(bargap=0.05)
    st.plotly_chart(fig_risk, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ④ 24-h scatter for selected parameter  (with spec lines)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"### Parameter Distribution — **{selected_param}** at 24 h")

p24 = param_df[(param_df["parameter"] == selected_param) & (param_df["time_h"] == 24.0)].copy()

if p24.empty:
    st.info("No 24 h data for this parameter in this lot.")
else:
    unit_label = p24["unit"].iloc[0] if not p24.empty else ""
    fig_scatter = px.scatter(
        p24,
        x="component_id",
        y="normalized_value",
        color="status",
        color_discrete_map=STATUS_COLOR,
        hover_data={
            "component_id":  True,
            "normalized_value": ":.4f",
            "status":        True,
            "risk_score":    ":.3f",
            "robust_z_lot":  ":.2f",
            "isolation_percentile": ":.1f",
            "mahalanobis_percentile": ":.1f",
        },
        title=f"24 h  {selected_param}  ({unit_label})",
        labels={"normalized_value": f"{selected_param} ({unit_label})"},
    )
    # Spec limit lines
    spec = SPEC_LIMITS.get(selected_param)
    if spec:
        if spec[0] in ("max", "two"):
            lim = spec[1] if spec[0] == "max" else spec[2]
            fig_scatter.add_hline(
                y=lim, line_dash="dash", line_color="#E74C3C",
                annotation_text=f"Spec MAX {lim}",
            )
        if spec[0] in ("min", "two"):
            fig_scatter.add_hline(
                y=spec[1], line_dash="dash", line_color="#E74C3C",
                annotation_text=f"Spec MIN {spec[1]}",
            )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ⑤ Drift trajectory  (time-series) for selected parameter
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"### Drift Trajectories — **{selected_param}** across all checkpoints")

param_ts = param_df[param_df["parameter"] == selected_param].copy()
SAMPLE_LIMIT = 40

all_comp_ids    = param_ts["component_id"].unique()
flagged_ids     = param_ts[param_ts["status"].isin(["QUARANTINE","STATIC_FAIL","MONITOR"])]["component_id"].unique()
non_flagged_ids = [c for c in all_comp_ids if c not in flagged_ids]
sample_ids      = list(flagged_ids) + non_flagged_ids[:max(0, SAMPLE_LIMIT - len(flagged_ids))]

ts_sample = param_ts[param_ts["component_id"].isin(sample_ids)].sort_values("time_h")
# Build per-component worst status for colouring
comp_status_map = (
    ts_sample.groupby("component_id")["status"]
    .agg(lambda s: max(s, key=lambda v: STATUS_PRIORITY.get(v, 0)))
    .to_dict()
)
ts_sample = ts_sample.copy()
ts_sample["line_color"] = ts_sample["component_id"].map(
    lambda c: STATUS_COLOR.get(comp_status_map.get(c, "NORMAL"), "#999")
)

fig_ts = go.Figure()
for comp_id, grp in ts_sample.groupby("component_id"):
    grp = grp.sort_values("time_h")
    worst = comp_status_map.get(comp_id, "NORMAL")
    color = STATUS_COLOR.get(worst, "#999")
    opacity = 0.9 if worst in ("QUARANTINE", "STATIC_FAIL") else 0.4
    fig_ts.add_trace(go.Scatter(
        x=grp["time_h"],
        y=grp["normalized_value"],
        mode="lines+markers",
        name=f"{comp_id} [{worst}]",
        line=dict(color=color, width=2 if worst in ("QUARANTINE","STATIC_FAIL","MONITOR") else 1),
        opacity=opacity,
        showlegend=worst in ("QUARANTINE", "STATIC_FAIL", "MONITOR"),
        hovertemplate=(
            f"<b>{comp_id}</b><br>"
            "Time: %{x}h<br>"
            "Value: %{y:.4f}<br>"
            f"Status: {worst}<extra></extra>"
        ),
    ))

spec = SPEC_LIMITS.get(selected_param)
if spec:
    if spec[0] in ("max", "two"):
        lim = spec[1] if spec[0] == "max" else spec[2]
        fig_ts.add_hline(y=lim, line_dash="dot", line_color="#E74C3C", annotation_text=f"Spec MAX {lim}")
    if spec[0] in ("min", "two"):
        fig_ts.add_hline(y=spec[1], line_dash="dot", line_color="#E74C3C", annotation_text=f"Spec MIN {spec[1]}")

unit_label = param_ts["unit"].iloc[0] if not param_ts.empty else ""
fig_ts.update_layout(
    title=f"Drift Trajectories — {selected_param} — {selected_lot}",
    xaxis_title="Burn-in Time (h)",
    yaxis_title=f"{selected_param} ({unit_label})",
    legend_title="Flagged Components",
)
st.plotly_chart(fig_ts, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ⑥ Multivariate scores heatmap (Isolation Forest + Mahalanobis)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Multivariate Anomaly Scores (All Parameters Together)")

mv_cols  = ["component_id", "parameter", "time_h",
            "isolation_percentile", "mahalanobis_percentile",
            "robust_z_lot", "slope_robust_z_lot", "status"]
mv_data  = param_df[mv_cols].dropna(subset=["isolation_percentile"])

col_iso, col_mah = st.columns(2)

with col_iso:
    st.markdown("#### Isolation Forest Percentile @ 24 h")
    iso24 = (
        param_df[param_df["time_h"] == 24.0]
        .pivot_table(index="component_id", columns="parameter",
                     values="isolation_percentile", aggfunc="first")
        .fillna(0)
    )
    if not iso24.empty:
        fig_iso = px.imshow(
            iso24.values,
            x=iso24.columns.tolist(),
            y=iso24.index.tolist(),
            color_continuous_scale="RdYlGn_r",
            zmin=0, zmax=100,
            aspect="auto",
            title="Isolation Forest Percentile (higher = more anomalous)",
            labels={"color": "Percentile"},
        )
        fig_iso.update_layout(height=400)
        fig_iso.add_hline(y=0, line_color="rgba(0,0,0,0)")  # spacer
        st.plotly_chart(fig_iso, use_container_width=True)
        st.caption("Components at 99.5th+ percentile are SEVERE; 97.5th+ are WARNING.")
    else:
        st.info("No isolation forest scores available at 24 h.")

with col_mah:
    st.markdown("#### Mahalanobis Percentile @ 24 h")
    mah24 = (
        param_df[param_df["time_h"] == 24.0]
        .pivot_table(index="component_id", columns="parameter",
                     values="mahalanobis_percentile", aggfunc="first")
        .fillna(0)
    )
    if not mah24.empty:
        fig_mah = px.imshow(
            mah24.values,
            x=mah24.columns.tolist(),
            y=mah24.index.tolist(),
            color_continuous_scale="RdYlGn_r",
            zmin=0, zmax=100,
            aspect="auto",
            title="Mahalanobis Percentile (higher = more anomalous)",
            labels={"color": "Percentile"},
        )
        fig_mah.update_layout(height=400)
        st.plotly_chart(fig_mah, use_container_width=True)
        st.caption("Mahalanobis detects unusual *combinations* of parameters.")
    else:
        st.info("No Mahalanobis scores available at 24 h.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Robust Z-score heatmap (lot-relative) across all parameters
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Robust Z-Score Heatmap (Lot-Relative) @ 24 h")

z_pivot = (
    param_df[param_df["time_h"] == 24.0]
    .pivot_table(index="component_id", columns="parameter",
                 values="robust_z_lot", aggfunc="first")
)
if not z_pivot.empty:
    fig_z = px.imshow(
        z_pivot.values,
        x=z_pivot.columns.tolist(),
        y=z_pivot.index.tolist(),
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        zmin=-6, zmax=6,
        aspect="auto",
        title="Robust Z-score vs. Lot Median (|Z|≥3.5 = SEVERE, |Z|≥5 = EXTREME)",
        labels={"color": "Z-score"},
    )
    fig_z.update_layout(height=420)
    st.plotly_chart(fig_z, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ⑧ Component Deep-Dive (audit panel)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Component Deep-Dive Audit")

flagged_comp_df = comp_df[comp_df["status"].isin(["QUARANTINE","STATIC_FAIL","MONITOR"])].copy()
all_comp_ids_sorted = sorted(comp_df["component_id"].unique())
flagged_ids_sorted  = sorted(flagged_comp_df["component_id"].tolist())

audit_mode = st.radio(
    "Audit scope",
    ["Flagged only (QUARANTINE / STATIC_FAIL / MONITOR)", "All components"],
    horizontal=True,
)
audit_pool = flagged_ids_sorted if "Flagged" in audit_mode else all_comp_ids_sorted

if not audit_pool:
    st.success("No flagged components in this lot.")
else:
    selected_comp = st.selectbox("Select Component ID", audit_pool)

    if selected_comp:
        comp_row = comp_df[comp_df["component_id"] == selected_comp].iloc[0]
        comp_param_rows = param_df[param_df["component_id"] == selected_comp].copy()

        # Status badge
        status_val  = comp_row["status"]
        badge_color = STATUS_COLOR.get(status_val, "#aaa")
        st.markdown(
            f"**Component `{selected_comp}`** — "
            f"<span style='background:{badge_color};color:white;padding:3px 10px;"
            f"border-radius:4px;font-weight:bold'>{status_val}</span>  "
            f"&nbsp; Risk score: **{comp_row['risk_score']:.3f}**  "
            f"&nbsp; Highest-risk param: **{comp_row['highest_risk_parameter']}** "
            f"@ **{comp_row['highest_risk_checkpoint_h']}h**",
            unsafe_allow_html=True,
        )

        # Reason codes
        reason_codes = comp_row["reason_codes"]
        if reason_codes:
            st.markdown(
                "**Reason codes:** "
                + "  ".join(
                    f"`{r}`" for r in reason_codes
                )
            )

        # ── Per-parameter tabs ────────────────────────────────────────────
        tab_labels  = sorted(comp_param_rows["parameter"].unique())
        audit_tabs  = st.tabs(tab_labels)

        for tab, param_name in zip(audit_tabs, tab_labels):
            with tab:
                prows = comp_param_rows[comp_param_rows["parameter"] == param_name].sort_values("time_h")
                unit  = prows["unit"].iloc[0] if not prows.empty else ""

                col_a, col_b = st.columns(2)

                # — Time-series drift —
                with col_a:
                    st.markdown(f"**Drift trajectory — {param_name}**")
                    fig_drift = go.Figure()
                    fig_drift.add_trace(go.Scatter(
                        x=prows["time_h"], y=prows["normalized_value"],
                        mode="lines+markers", name=selected_comp,
                        line=dict(color=badge_color, width=2),
                        hovertemplate="Time: %{x}h<br>Value: %{y:.4f}<extra></extra>",
                    ))
                    spec = SPEC_LIMITS.get(param_name)
                    if spec:
                        if spec[0] in ("max","two"):
                            lim = spec[1] if spec[0]=="max" else spec[2]
                            fig_drift.add_hline(y=lim, line_dash="dot", line_color="red",
                                                annotation_text=f"Spec MAX {lim}")
                        if spec[0] in ("min","two"):
                            fig_drift.add_hline(y=spec[1], line_dash="dot", line_color="red",
                                                annotation_text=f"Spec MIN {spec[1]}")
                    fig_drift.update_layout(
                        xaxis_title="Burn-in (h)",
                        yaxis_title=f"{param_name} ({unit})",
                        height=300,
                        margin=dict(t=20, b=20),
                    )
                    st.plotly_chart(fig_drift, use_container_width=True)

                # — Lot distribution histogram at each checkpoint —
                with col_b:
                    st.markdown(f"**Lot distribution** — `{selected_comp}` vs. lot")
                    lot_param_all = param_df[param_df["parameter"] == param_name]
                    for t in sorted(prows["time_h"].unique()):
                        lot_t    = lot_param_all[lot_param_all["time_h"] == t]["normalized_value"].dropna()
                        comp_val = prows[prows["time_h"] == t]["normalized_value"]
                        if lot_t.empty or comp_val.empty:
                            continue
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Histogram(
                            x=lot_t, name="Lot", opacity=0.6,
                            marker_color="#3498DB", nbinsx=20,
                        ))
                        fig_hist.add_vline(
                            x=float(comp_val.values[0]),
                            line_dash="dash", line_color=badge_color,
                            annotation_text=f"{t}h: {float(comp_val.values[0]):.3f}",
                        )
                        fig_hist.update_layout(
                            title_text=f"{t}h checkpoint",
                            xaxis_title=f"{param_name} ({unit})",
                            height=190, margin=dict(l=20, r=20, t=30, b=20),
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                # — Detailed score table for this parameter —
                score_cols = [
                    "time_h", "normalized_value", "static_status", "static_margin",
                    "robust_z_lot", "robust_z_historical",
                    "lot_percentile", "historical_percentile",
                    "slope", "slope_robust_z_lot", "slope_robust_z_historical",
                    "isolation_score", "isolation_percentile",
                    "mahalanobis_score", "mahalanobis_percentile",
                    "risk_score", "status",
                ]
                available_score_cols = [c for c in score_cols if c in prows.columns]
                score_tbl = prows[available_score_cols].sort_values("time_h").reset_index(drop=True)
                for float_col in ["normalized_value","static_margin","robust_z_lot",
                                  "robust_z_historical","lot_percentile","historical_percentile",
                                  "slope","slope_robust_z_lot","slope_robust_z_historical",
                                  "isolation_score","isolation_percentile",
                                  "mahalanobis_score","mahalanobis_percentile","risk_score"]:
                    if float_col in score_tbl.columns:
                        score_tbl[float_col] = score_tbl[float_col].apply(
                            lambda v: f"{v:.4f}" if pd.notna(v) else "—"
                        )
                st.markdown("**Detailed scores per checkpoint**")
                st.dataframe(score_tbl, use_container_width=True, hide_index=True)

                # — Reason codes for this parameter —
                param_reasons = [
                    r for row in prows.to_dict("records")
                    for r in (row.get("reason_codes") or [])
                ]
                if param_reasons:
                    st.markdown("**Reason codes:** " + "  ".join(f"`{r}`" for r in sorted(set(param_reasons))))

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ⑨ Full component summary table
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Full Component Summary Table")

summary_cols = [
    "component_id", "status", "risk_score",
    "highest_risk_parameter", "highest_risk_checkpoint_h",
]
summary = comp_df[summary_cols].copy()
summary["risk_score"] = summary["risk_score"].map(lambda v: f"{v:.4f}")
summary = summary.sort_values(
    "status", key=lambda s: s.map(STATUS_PRIORITY), ascending=False
).reset_index(drop=True)

def _color_status_cell(val: str) -> str:
    return f"background-color:{STATUS_COLOR.get(val,'')};color:white;font-weight:bold"

st.dataframe(
    summary.style.map(_color_status_cell, subset=["status"]),
    use_container_width=True,
    height=400,
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ⑩ Cross-lot comparison  (risk score trends across all lots)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Cross-Lot Comparison — Risk Score Trends")

if st.button("Compute Cross-Lot Trend (Warning: Takes ~2-3 minutes to score all lots)"):
    with st.spinner("Scoring all lots in the background..."):
        all_lots = {l_id: score_single_lot(id(engine), l_id) for l_id in available_lots}
        cross_rows = []
    for lot_id, result in all_lots.items():
        if "error" in result:
            continue
        cdf = pd.DataFrame(result.get("component_results", []))
        if cdf.empty:
            continue
        for status in ["NORMAL","MONITOR","QUARANTINE","STATIC_FAIL","RETEST_REQUIRED"]:
            cross_rows.append({
                "lot_id": lot_id,
                "status": status,
                "count":  int((cdf["status"] == status).sum()),
            })

    cross_df = pd.DataFrame(cross_rows)
    if not cross_df.empty:
        fig_cross = px.bar(
            cross_df,
            x="lot_id", y="count", color="status",
            color_discrete_map=STATUS_COLOR,
            barmode="stack",
            title="Component Status by Lot (all lots)",
            labels={"count": "Component Count", "lot_id": "Lot ID"},
        )
        fig_cross.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_cross, use_container_width=True)

    # Average risk score trend
    avg_risk_rows = []
    for lot_id, result in all_lots.items():
        if "error" in result:
            continue
        cdf = pd.DataFrame(result.get("component_results", []))
        if cdf.empty:
            continue
        avg_risk_rows.append({
            "lot_id":       lot_id,
            "avg_risk":     cdf["risk_score"].mean(),
            "max_risk":     cdf["risk_score"].max(),
            "p95_risk":     cdf["risk_score"].quantile(0.95),
        })

    avg_risk_df = pd.DataFrame(avg_risk_rows).sort_values("lot_id")
    if not avg_risk_df.empty:
        fig_avgr = go.Figure()
        fig_avgr.add_trace(go.Scatter(
            x=avg_risk_df["lot_id"], y=avg_risk_df["avg_risk"],
            mode="lines+markers", name="Mean Risk", line=dict(color="#3498DB"),
        ))
        fig_avgr.add_trace(go.Scatter(
            x=avg_risk_df["lot_id"], y=avg_risk_df["p95_risk"],
            mode="lines+markers", name="95th Pct Risk", line=dict(color="#E74C3C", dash="dot"),
        ))
        fig_avgr.update_layout(
            title="Risk Score Trend Across Lots",
            xaxis_title="Lot ID",
            yaxis_title="Risk Score",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_avgr, use_container_width=True)

st.divider()
st.caption(
    "ESS QA Inspector · ess_module_a v0.1.0 · "
    "Classification via Isolation Forest + Mahalanobis + Robust Z + IQR + Static Spec checks."
)
