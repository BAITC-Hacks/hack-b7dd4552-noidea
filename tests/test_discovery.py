import json
from datetime import date
from urllib.parse import parse_qs, urlsplit

import pytest

from wind import discovery, sources
from wind.discovery import (
    Decision,
    DiscoveryInput,
    QueryPlan,
    build_config,
    context_for,
    discover,
    load_next,
    ranges,
    validate_coverage,
    within_site,
)
from wind.sources import Mapping
from wind.storage import all_metadata, create_turbine, save_metadata


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    t = create_turbine({"name": "Test", "latitude": 43.64, "longitude": 78.53})
    save_metadata(
        "datasets",
        t["id"],
        {
            "id": t["id"],
            "sha256": "abc",
            "rows": 5000,
            "start": "2026-01-01T00:00:00",
            "end": "2026-02-02T23:50:00",
        },
    )
    request = DiscoveryInput(site="weather.example", turbine_id=t["id"])
    page = {
        "url": request.site,
        "text": "Public endpoint https://api.weather.example/data",
        "links": ["https://weather.example/docs"],
        "sha256": "test",
    }
    plan = QueryPlan(
        endpoint="https://api.weather.example/data",
        params={"units": "ms"},
        latitude_param="lat",
        longitude_param="lon",
        start_param="from",
        end_param="to",
        mapping=Mapping(
            time_field="time",
            wind_field="wind",
            timezone="UTC",
            wind_unit="m/s",
            wind_height_m=100,
            timestamp_semantics="instant",
        ),
        evidence_url=request.site,
        data_kind="history",
        explanation="Документированная погода",
    )
    monkeypatch.setattr(discovery, "read_page", lambda url: page | {"url": url})
    monkeypatch.setattr(
        discovery,
        "ask",
        lambda messages: (Decision(action="propose", plan=plan), {"input_tokens": 10}),
    )

    def download(config):
        params = parse_qs(urlsplit(config.url).query)
        times = discovery.pd_range(
            date.fromisoformat(params["from"][0]), date.fromisoformat(params["to"][0])
        )
        return json.dumps([{"time": time, "wind": 4} for time in times]).encode()

    monkeypatch.setattr(discovery, "download", download)
    monkeypatch.setattr(sources, "download", download)
    return request, page, plan


def test_scope_and_bad_sites():
    assert within_site("https://api.weather.example/data", "https://weather.example")
    assert not within_site("https://weather.example.evil.org/", "https://weather.example")
    assert not within_site("https://evilweather.example/", "https://weather.example")
    assert not within_site("http://weather.example/", "https://weather.example")
    with pytest.raises(ValueError):
        DiscoveryInput(site="file:///etc/passwd", turbine_id=1)


def test_server_overrides_coordinates_dates_and_requires_documented_endpoint(setup):
    req, page, plan = setup
    plan.params.update(lat="0", lon="0", **{"from": "1900-01-01"})
    cfg = build_config(plan, req, context_for(req), date(2026, 1, 1), date(2026, 1, 2), [page])
    query = parse_qs(urlsplit(cfg.url).query)
    assert query["lat"] == ["43.64"] and query["from"] == ["2026-01-01"]
    plan.endpoint = "https://api.weather.example/undocumented"
    with pytest.raises(ValueError, match="Endpoint"):
        build_config(plan, req, context_for(req), date(2026, 1, 1), date(2026, 1, 2), [page])


def test_padded_chunks_no_gaps(setup):
    req, _, _ = setup
    chunks = ranges(context_for(req), "history")
    assert chunks == [
        (date(2025, 12, 31), date(2026, 1, 30)),
        (date(2026, 1, 31), date(2026, 2, 3)),
    ]


def test_discovery_and_resume_no_duplicate_completion(setup):
    req, _, _ = setup
    result = discover(req)
    assert result["status"] == "ready" and len(result["chunks"]) == 2
    assert all_metadata("sources") == []
    first = load_next(result["id"])
    assert len(first["completed"]) == 1 and first["status"] == "ready"
    done = load_next(result["id"])
    assert done["status"] == "complete" and len(done["completed"]) == 2
    assert load_next(result["id"])["completed"] == done["completed"]
    assert len(all_metadata("sources")) == 1


def test_incomplete_coverage_never_saved(setup, monkeypatch):
    req, _, _ = setup
    result = discover(req)
    monkeypatch.setattr(sources, "download", lambda cfg: b'[{"time":"2025-12-31T00:00Z","wind":4}]')
    with pytest.raises(ValueError, match="период"):
        load_next(result["id"])
    assert not (sources.data_dir() / "source-runs").exists()
    with pytest.raises(ValueError):
        validate_coverage([], date(2026, 1, 1), date(2026, 1, 1))


def test_changed_dataset_or_disabled_source_stops_resume(setup):
    req, _, _ = setup
    report = discover(req)
    first = load_next(report["id"])
    source = sources.get_source(first["source_id"])
    save_metadata("sources", source["id"], source | {"enabled": False})
    with pytest.raises(ValueError, match="отключён"):
        load_next(report["id"])
    save_metadata("sources", source["id"], source)
    dataset = all_metadata("datasets")[0]
    save_metadata("datasets", dataset["id"], dataset | {"sha256": "changed"})
    with pytest.raises(ValueError, match="изменились"):
        load_next(report["id"])


def test_purpose_mismatch_and_unresolved_block(setup):
    req, page, plan = setup
    plan.data_kind = "historical_forecast"
    with pytest.raises(ValueError, match="Цель"):
        build_config(plan, req, context_for(req), date(2026, 1, 1), date(2026, 1, 1), [page])
    plan.data_kind = "history"
    plan.unresolved = ["Timezone?"]
    with pytest.raises(ValueError, match="Timezone"):
        build_config(plan, req, context_for(req), date(2026, 1, 1), date(2026, 1, 1), [page])


def test_unknown_source_stops_with_explanation(setup, monkeypatch):
    req, _, _ = setup
    monkeypatch.setattr(
        discovery, "ask", lambda m: (Decision(action="stop", explanation="Нужен API-ключ"), {})
    )
    result = discover(req)
    assert result["status"] == "needs_input" and not result["chunks"]
    assert "ключ" in result["explanation"]


def test_only_found_documentation_links_can_be_read(setup, monkeypatch):
    req, _, _ = setup
    calls = []

    def ask(messages):
        calls.append(1)
        return Decision(action="read", url="https://weather.example/secret"), {}

    monkeypatch.setattr(discovery, "ask", ask)
    result = discover(req)
    assert len(calls) == 6 and len(result["documents"]) == 1
    assert all(not step["ok"] for step in result["steps"])


def test_invalid_llm_response_can_be_repaired(setup, monkeypatch):
    req, _, plan = setup
    attempts = []

    def ask(messages):
        attempts.append(1)
        if len(attempts) == 1:
            raise discovery.DecisionError("timezone auto invalid", "{}", {"input_tokens": 7})
        return Decision(action="propose", plan=plan), {"input_tokens": 3}

    monkeypatch.setattr(discovery, "ask", ask)
    result = discover(req)
    assert result["status"] == "ready" and result["input_tokens"] == 10
    assert result["steps"][0]["action"] == "validate_plan"


def test_openmeteo_reanalysis_is_not_a_historical_forecast(setup):
    req, page, plan = setup
    req.site = "https://open-meteo.com"
    req.purpose = "historical_forecast"
    plan.data_kind = "historical_forecast"
    plan.endpoint = "https://archive-api.open-meteo.com/v1/archive"
    with pytest.raises(ValueError, match="другой цели"):
        build_config(plan, req, context_for(req), date(2026, 1, 1), date(2026, 1, 1), [page])


def test_forecast_uses_today_instead_of_csv_dates(setup):
    from datetime import UTC, datetime, timedelta

    req, _, _ = setup
    today = datetime.now(UTC).date()
    assert ranges(context_for(req), "forecast") == [(today, today + timedelta(days=1))]
