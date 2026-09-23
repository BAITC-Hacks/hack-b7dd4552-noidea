import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from wind import turbine_specs as specs
from wind.storage import create_turbine, delete_turbine, get_turbine, save_metadata


def node(node_id=9690012914, **changes):
    return {
        "type": "node",
        "id": node_id,
        "lat": 43.6452141,
        "lon": 78.5355410,
        "tags": {
            "power": "generator",
            "generator:source": "wind",
            "generator:output:electricity": "2500 kW",
            "manufacturer": "Goldwind",
            "model": "GW109/2500",
        },
        **changes,
    }


@pytest.fixture
def turbine(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    specs._CACHE.clear()
    return create_turbine({"name": "T1", "latitude": 43.64515, "longitude": 78.535604})


@pytest.fixture
def mock_http(monkeypatch):
    original = httpx.Client

    def install(handler):
        monkeypatch.setattr(
            specs.httpx,
            "Client",
            lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(handler)),
        )

    return install


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2500 kW", 2500),
        ("2.5 MW", 2500),
        ("2,5MW", 2500),
        ("2500", None),
        ("yes", None),
        ("2.5 GW", None),
        ("0 kW", None),
        (None, None),
        ("NaN kW", None),
        ("2.5 MW;3 MW", None),
    ],
)
def test_capacity_requires_explicit_units(value, expected):
    assert specs.parse_capacity(value) == expected


def test_lookup_is_bounded_deduplicates_and_does_not_apply(turbine, mock_http):
    calls = []

    def handler(request):
        calls.append(request)
        assert str(request.url).startswith(specs.OVERPASS_URL)
        assert "around:100,43.64515,78.535604" in request.url.params["data"]
        return httpx.Response(
            200,
            json={
                "elements": [
                    node(),
                    node(),
                    node(2, lat=40),
                    node(3, tags={"power": "generator", "generator:source": "wind"}),
                ]
            },
        )

    mock_http(handler)
    result = specs.get_specifications(turbine["id"])
    assert len(result["candidates"]) == 2
    assert result["candidates"][0]["distance_m"] == 8.75
    assert result["candidates"][1]["rated_power_kw"] is None
    assert result["current"]["rated_power_kw"] is None
    assert result["confirmation_required"] is True
    assert get_turbine(turbine["id"]) == turbine
    specs.get_specifications(turbine["id"])
    assert len(calls) == 1


def xml_node(node_id=9690012914, capacity="2.5 MW", lat=43.6452141):
    return (
        f'<osm><node id="{node_id}" lat="{lat}" lon="78.535541">'
        '<tag k="power" v="generator"/><tag k="generator:source" v="wind"/>'
        f'<tag k="generator:output:electricity" v="{capacity}"/>'
        '<tag k="manufacturer" v="Goldwind"/><tag k="model" v="GW109/2500"/>'
        "</node></osm>"
    )


def test_explicit_confirmation_fetches_fixed_node_and_preserves_catalog(turbine, mock_http):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, text=xml_node())

    mock_http(handler)
    result = specs.confirm_specifications(
        turbine["id"], specs.ConfirmSpecifications(osm_node_id=9690012914)
    )
    assert calls == [specs.OSM_API + "9690012914"]
    saved = result["turbine"]
    assert all(saved[key] == value for key, value in turbine.items())
    assert saved["rated_power_kw"] == 2500
    assert saved["capacity_status"] == "user_confirmed_osm"
    assert saved["capacity_source_url"] == "https://www.openstreetmap.org/node/9690012914"
    assert saved["turbine_model"] == "GW109/2500"


@pytest.mark.parametrize("xml", [xml_node(10), xml_node(capacity="yes"), xml_node(lat=40)])
def test_other_node_unknown_capacity_or_distant_equipment_cannot_be_applied(
    turbine, mock_http, xml
):
    mock_http(lambda request: httpx.Response(200, text=xml))
    with pytest.raises(HTTPException) as error:
        specs.confirm_specifications(
            turbine["id"], specs.ConfirmSpecifications(osm_node_id=9690012914)
        )
    assert error.value.status_code == 400
    assert get_turbine(turbine["id"]) == turbine


def test_deleted_during_network_lookup_is_not_restored(turbine, mock_http):
    def handler(request):
        delete_turbine(turbine["id"])
        return httpx.Response(200, text=xml_node())

    mock_http(handler)
    with pytest.raises(HTTPException) as error:
        specs.confirm_specifications(
            turbine["id"], specs.ConfirmSpecifications(osm_node_id=9690012914)
        )
    assert error.value.status_code == 404


def test_coordinates_changed_during_confirmation_are_rechecked(turbine, mock_http):
    def handler(request):
        save_metadata("turbines", turbine["id"], {**turbine, "latitude": 40})
        return httpx.Response(200, text=xml_node())

    mock_http(handler)
    with pytest.raises(HTTPException) as error:
        specs.confirm_specifications(
            turbine["id"], specs.ConfirmSpecifications(osm_node_id=9690012914)
        )
    assert error.value.status_code == 400
    assert get_turbine(turbine["id"])["latitude"] == 40


def test_sources_are_not_user_supplied_and_redirects_are_rejected(turbine, mock_http):
    for value in ("https://127.0.0.1/", "9690012914", True, -1):
        with pytest.raises(ValidationError):
            specs.ConfirmSpecifications(osm_node_id=value)
    with pytest.raises(ValidationError):
        specs.ConfirmSpecifications(osm_node_id=9690012914, source_url="http://localhost")
    mock_http(lambda request: httpx.Response(302, headers={"location": "http://127.0.0.1/"}))
    with pytest.raises(HTTPException) as error:
        specs.confirm_specifications(
            turbine["id"], specs.ConfirmSpecifications(osm_node_id=9690012914)
        )
    assert error.value.status_code == 502


def test_known_assignment_fallback_is_fetched_and_still_only_a_candidate(turbine, mock_http):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if request.url.host == "overpass-api.de":
            return httpx.Response(503)
        return httpx.Response(200, text=xml_node())

    mock_http(handler)
    result = specs.get_specifications(turbine["id"])
    assert len(calls) == 2
    assert result["warning"]
    assert result["candidates"][0]["rated_power_kw"] == 2500
    assert "rated_power_kw" not in get_turbine(turbine["id"])


def test_deleted_turbine_does_not_make_network_call(turbine, mock_http):
    delete_turbine(turbine["id"])
    mock_http(lambda request: pytest.fail("deleted turbine should not trigger lookup"))
    with pytest.raises(HTTPException) as error:
        specs.get_specifications(turbine["id"])
    assert error.value.status_code == 404
