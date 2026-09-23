import json

import pytest
from fastapi.testclient import TestClient

from wind import sources
from wind.app import app
from wind.sources import Mapping, SourceInput, normalize, parse


def config(**changes):
    values = dict(
        name="Weather",
        url="https://weather.example/data",
        trusted=True,
        mapping=Mapping(
            time_field="time",
            wind_field="wind",
            timezone="Etc/GMT-6",
            wind_unit="km/h",
            wind_height_m=100,
            timestamp_semantics="instant",
        ),
    )
    return SourceInput(**(values | changes))


def test_csv_units_timezone_and_temperature():
    source = config(format="csv")
    source.mapping.temperature_field = "temp"
    source.mapping.temperature_unit = "K"
    rows = parse(b"time,wind,temp\n2026-01-01T06:00,36,273.15\n", source)
    assert normalize(rows, source.mapping) == [
        {"time": "2026-01-01T00:00:00+00:00", "wind_speed": 10, "temperature": 0}
    ]


def test_json_columns_and_schema_drift():
    source = config()
    source.mapping.rows_path = "hourly"
    raw = b'{"hourly":{"time":["2026-01-01T06:00"],"wind":[36]}}'
    assert normalize(parse(raw, source), source.mapping)[0]["wind_speed"] == 10
    with pytest.raises(ValueError):
        parse(b'{"hourly":{"time":[],"wind":[36]}}', source)
    with pytest.raises(ValueError):
        normalize([{"time": "2026-01-01T06:00", "new_wind": 36}], source.mapping)


@pytest.mark.parametrize("wind", [-1, "NaN", "inf", None, True])
def test_bad_values_rejected(wind):
    with pytest.raises(ValueError):
        normalize([{"time": "2026-01-01T06:00", "wind": wind}], config().mapping)


def test_time_missing_ambiguous_conflicting_and_duplicates():
    mapping = config().mapping
    row = {"time": "2026-01-01T06:00", "wind": 36}
    with pytest.raises(ValueError):
        normalize([row, row], mapping)
    with pytest.raises(ValueError):
        normalize([row | {"time": "2026-01-01T06:00Z"}], mapping)
    mapping.timezone = ""
    with pytest.raises(ValueError):
        normalize([row], mapping)
    mapping.timezone = "Asia/Almaty"
    with pytest.raises(ValueError):
        normalize([row | {"time": "2024-02-29T23:30"}], mapping)


@pytest.mark.parametrize("address", ["127.0.0.1", "169.254.169.254", "10.0.0.1", "::1"])
def test_private_addresses_blocked(monkeypatch, address):
    monkeypatch.setattr(
        sources.socket, "getaddrinfo", lambda *a, **k: [(0, 0, 0, "", (address, 443))]
    )
    with pytest.raises(ValueError, match="адреса"):
        sources.download(config())


def test_disabled_and_untrusted_never_fetch(monkeypatch):
    def fail(*a, **k):
        raise AssertionError("DNS must not be used")

    monkeypatch.setattr(sources.socket, "getaddrinfo", fail)
    for c in [config(trusted=False), config(enabled=False)]:
        with pytest.raises(ValueError):
            sources.download(c)


def test_registry_edit_preview_load_and_failed_import(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(sources, "download", lambda c: b'[{"time":"2026-01-01T06:00","wind":36}]')
    with TestClient(app) as client:
        assert client.get("/api/sources").json() == {"items": []}
        created = client.post("/api/sources", json=config().model_dump()).json()
        base = "/api/sources/" + created["id"]
        assert client.post(base + "/preview").json()["rows"] == 1
        assert not (tmp_path / "source-runs").exists()
        loaded = client.post(base + "/load").json()
        assert (
            client.get("/api/sources/batches/" + loaded["batch_id"]).json()["points"][0][
                "wind_speed"
            ]
            == 10
        )
        updated = config(name="Edited").model_dump()
        assert client.put(base, json=updated).json()["revision"] == 2
        monkeypatch.setattr(sources, "download", lambda c: b'[{"time":"bad","wind":36}]')
        assert client.post(base + "/load").status_code == 422
        assert len(list((tmp_path / "source-runs").iterdir())) == 1


def test_suggestion_is_validated_not_saved(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    monkeypatch.setattr(sources, "download", lambda c: b'[{"time":"2026-01-01T06:00","wind":36}]')
    output = {"mapping": config().mapping.model_dump(), "explanation": "Example", "unresolved": []}

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, **kwargs):
            return sources.httpx.Response(
                200,
                json={
                    "status": "completed",
                    "output": [{"content": [{"type": "output_text", "text": json.dumps(output)}]}],
                },
            )

    monkeypatch.setattr(sources.httpx, "Client", Client)
    result = sources.suggest(config())
    assert result["validation"]["ok"] and not result["saved"]
    output["mapping"]["wind_field"] = "invented"
    assert not sources.suggest(config())["validation"]["ok"]
    assert not (tmp_path / "source-runs").exists()


@pytest.mark.parametrize(
    "status,raw,error",
    [(302, b"", "HTTP 302"), (200, b"abc", None), (200, b"x" * (sources.LIMIT + 1), "2 МБ")],
)
def test_pinned_download_redirects_and_size(monkeypatch, status, raw, error):
    import io

    calls = []
    monkeypatch.setattr(
        sources.socket, "getaddrinfo", lambda *a, **k: [(0, 0, 0, "", ("8.8.8.8", 443))]
    )
    monkeypatch.setattr(sources.socket, "create_connection", lambda addr, *a: calls.append(addr))

    class Connection:
        sock = None

        def __init__(self, host, **kwargs):
            assert host == "weather.example"

        def request(self, method, target, **kwargs):
            self._create_connection(("weather.example", 443), 20)

        def getresponse(self):
            result = io.BytesIO(raw)
            result.status = status
            return result

        def close(self):
            pass

    monkeypatch.setattr(sources.http.client, "HTTPSConnection", Connection)
    if error:
        with pytest.raises(ValueError, match=error):
            sources.download(config())
    else:
        assert sources.download(config()) == raw
    assert calls == [("8.8.8.8", 443)]


def test_reported_units_cannot_silently_disagree():
    source = config()
    source.mapping.rows_path = "hourly"
    raw = b'{"hourly_units":{"wind":"m/s"},"hourly":{"time":["2026-01-01T06:00"],"wind":[36]}}'
    with pytest.raises(ValueError, match="единицы ветра"):
        parse(raw, source)
    source.mapping.wind_unit = "m/s"
    assert normalize(parse(raw, source), source.mapping)[0]["wind_speed"] == 36
