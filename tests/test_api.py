from __future__ import annotations

from fastapi.testclient import TestClient

from ess_module_a.api import create_app


def test_health_and_model_info(trained_engine):
    client = TestClient(create_app(trained_engine))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["reference_loaded"] is True
    info = client.get("/v1/module-a/model-info")
    assert info.status_code == 200


def test_score_endpoint(trained_engine, test_lot):
    client = TestClient(create_app(trained_engine))
    records = test_lot.loc[test_lot["time_h"] <= 24].to_dict("records")
    response = client.post(
        "/v1/module-a/score-lot",
        json={"measurements": records, "as_of_h": 24},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["lot_id"] == "TEST_LOT_001"
    assert len(payload["component_results"]) == 40
