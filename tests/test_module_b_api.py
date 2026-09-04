from __future__ import annotations

from fastapi.testclient import TestClient

from ess_module_b.api import create_app


def test_module_b_health_model_info_and_forecast(trained_module_b_engine, test_lot):
    client = TestClient(create_app(trained_module_b_engine))
    assert client.get("/health").json()["artifact_loaded"] is True
    info = client.get("/v1/module-b/model-info")
    assert info.status_code == 200
    assert info.json()["context_model_count"] > 0
    records = test_lot.loc[test_lot["time_h"] <= 24].to_dict("records")
    response = client.post(
        "/v1/module-b/forecast-lot", json={"measurements": records}
    )
    assert response.status_code == 200, response.text
    assert response.json()["target_h"] == 168.0
    assert len(response.json()["component_results"]) == 40
