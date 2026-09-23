import json
from datetime import UTC, datetime, timedelta

import httpx
import numpy as np
import pytest
import xarray as xr

from wind import archive
from wind.storage import create_turbine, delete_turbine

RUN = datetime(2026, 1, 31, tzinfo=UTC)
ISSUE = RUN + timedelta(hours=12)


def listing(run, *, missing=(), changed=None):
    entries = []
    for lead in range(97):
        if lead in missing:
            continue
        published = (changed or {}).get(lead, run + timedelta(hours=4))
        entries.append(
            f"<Contents><Key>{archive._prefix(run)}{lead:03}</Key>"
            f"<LastModified>{published.isoformat()}</LastModified>"
            '<ETag>"0123456789abcdef0123456789abcdef"</ETag><Size>500000000</Size></Contents>'
        )
    return (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        "<IsTruncated>false</IsTruncated>" + "".join(entries) + "</ListBucketResult>"
    ).encode()


def raw_point(run, latitude, longitude):
    return {
        "run": run.isoformat(),
        "grid_latitude": latitude,
        "grid_longitude": longitude,
        "dataset_snapshot": archive.DATASET_SNAPSHOT,
        "points": [
            {
                "time": (run + timedelta(hours=h)).isoformat(),
                "lead_hour": h,
                "temperature_2m": -2.0,
                "wind_u_10m": 3.0,
                "wind_v_10m": 4.0,
                "wind_u_100m": -3.0,
                "wind_v_100m": -4.0,
            }
            for h in range(97)
        ],
    }


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    one = create_turbine({"name": "One", "latitude": 43.643198, "longitude": 78.538828})
    two = create_turbine({"name": "Two", "latitude": 43.64515, "longitude": 78.535604})
    data = {
        "one": one["id"],
        "two": two["id"],
        "requests": [],
        "points": [],
        "listing": listing,
        "point": raw_point,
    }

    def handle(request):
        assert request.url.host == "noaa-gfs-bdp-pds.s3.amazonaws.com"
        assert request.url.params["list-type"] == "2"
        prefix = request.url.params["prefix"]
        run = datetime.strptime(prefix.split("/")[0], "gfs.%Y%m%d").replace(tzinfo=UTC)
        data["requests"].append(run)
        return httpx.Response(200, content=data["listing"](run))

    real_client = httpx.Client
    monkeypatch.setattr(
        archive.httpx,
        "Client",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(handle),
            **kwargs,
        ),
    )

    def download(run, latitude, longitude):
        data["points"].append((run, latitude, longitude))
        return data["point"](run, latitude, longitude)

    monkeypatch.setattr(archive, "_download_point", download)
    return data


@pytest.mark.parametrize("horizon", [24, 48])
def test_exact_hourly_horizon_and_verified_source_provenance(context, horizon):
    result = archive.fetch_issue_weather(context["one"], ISSUE, horizon)
    assert result["eligible"] is True
    assert result["historical_eligibility"] == "verified"
    assert result["run"] == RUN.isoformat()
    assert result["available_at"] == (RUN + timedelta(hours=4)).isoformat()
    assert len(result["points"]) == horizon
    assert result["points"][0]["time"] == (ISSUE + timedelta(hours=1)).isoformat()
    assert result["points"][-1]["time"] == (ISSUE + timedelta(hours=horizon)).isoformat()
    point = result["points"][0]
    assert point["wind_speed_10m"] == point["wind_speed_100m"] == 5
    assert point["wind_direction_100m"] == pytest.approx(36.86989765)
    assert result["provenance"]["dataset_snapshot"] == archive.DATASET_SNAPSHOT
    assert len(result["provenance"]["source_objects"]) == horizon
    assert (
        result["provenance"]["availability_scope"] == "original_noaa_grib_objects_not_dynamical_api"
    )


def test_init_time_is_not_availability_and_late_current_run_falls_back(context):
    issue = RUN + timedelta(hours=3)
    result = archive.fetch_issue_weather(context["one"], issue)
    assert result["run"] == (RUN - timedelta(days=1)).isoformat()
    # No NWP bytes are downloaded for a run that was not available at issue.
    assert [run for run, _, _ in context["points"]] == [RUN - timedelta(days=1)]
    assert result["points"][0]["time"] == (issue + timedelta(hours=1)).isoformat()


def test_missing_or_backfilled_grib_is_not_given_schedule_based_eligibility(context):
    def late(run):
        return listing(
            run, changed={13: ISSUE + timedelta(seconds=1), 37: ISSUE + timedelta(seconds=1)}
        )

    context["listing"] = late
    with pytest.raises(archive.ArchiveUnavailableError, match="позже момента"):
        archive.fetch_issue_weather(context["one"], ISSUE)
    assert context["points"] == []


def test_source_published_at_cutoff_is_allowed_and_unused_future_file_is_ignored(context):
    context["listing"] = lambda run: listing(
        run, changed={60: ISSUE, 90: ISSUE + timedelta(days=1)}
    )
    result = archive.fetch_issue_weather(context["one"], ISSUE)
    assert result["available_at"] == ISSUE.isoformat()
    assert result["run"] == RUN.isoformat()


def test_missing_one_required_lead_blocks_whole_run_instead_of_mixing(context):
    context["listing"] = lambda run: listing(run, missing=(30,)) if run == RUN else listing(run)
    result = archive.fetch_issue_weather(context["one"], ISSUE)
    assert {p["run"] for p in result["points"]} == {(RUN - timedelta(days=1)).isoformat()}


def test_explicit_previous_run_and_utc_offset_normalization(context):
    result = archive.fetch_issue_weather(
        context["one"],
        "2026-01-31T18:00:00+06:00",
        previous_run=True,
    )
    assert result["run"] == (RUN - timedelta(days=1)).isoformat()
    assert result["issue_at"] == ISSUE.isoformat()
    assert context["requests"] == [RUN - timedelta(days=1)]


def test_nearby_turbines_reuse_only_actual_same_grid_cache_offline(context):
    first = archive.fetch_issue_weather(context["one"], ISSUE)
    second = archive.fetch_issue_weather(context["two"], ISSUE, offline=True)
    assert first["sha256"] == second["sha256"]
    assert first["points"] == second["points"]
    assert first["provenance"]["latitude"] != second["provenance"]["latitude"]
    assert len(context["requests"]) == len(context["points"]) == 1
    assert context["points"][0][1:] == (43.75, 78.5)


def test_unknown_or_deleted_turbine_does_not_fetch(context):
    delete_turbine(context["one"])
    for turbine_id in (context["one"], 99999):
        with pytest.raises(ValueError):
            archive.fetch_issue_weather(turbine_id, ISSUE)
    assert not context["requests"]


@pytest.mark.parametrize("horizon", [0, 23, 49, True, 48.0])
def test_unsupported_horizon_is_rejected_before_network(context, horizon):
    with pytest.raises(ValueError, match="Горизонт"):
        archive.fetch_issue_weather(context["one"], ISSUE, horizon)
    assert not context["requests"]


@pytest.mark.parametrize(
    "issue",
    ["2026-01-31T12:00:00", "2026-01-31T12:30:00Z", "2099-01-01T12:00:00Z", "2020-01-01T12:00:00Z"],
)
def test_ambiguous_off_grid_or_unsupported_issue_is_rejected(context, issue):
    with pytest.raises(ValueError):
        archive.fetch_issue_weather(context["one"], issue)
    assert not context["requests"]


def test_publication_listing_rejects_partial_duplicate_or_impossible_evidence():
    content = listing(RUN)
    invalid = [
        content.replace(b"<IsTruncated>false", b"<IsTruncated>true"),
        content.replace(b"</ListBucketResult>", content[content.index(b"<Contents>") :]),
        listing(RUN, changed={1: RUN - timedelta(seconds=1)}),
        content.replace(b"<Size>500000000</Size>", b"<Size>0</Size>"),
    ]
    for payload in invalid:
        with pytest.raises(archive.ArchiveUnavailableError):
            archive.parse_publication_listing(payload, RUN)


def test_missing_nwp_value_does_not_become_zero_or_future_fill(context):
    def missing(run, latitude, longitude):
        data = raw_point(run, latitude, longitude)
        data["points"][40]["wind_u_100m"] = None
        return data

    context["point"] = missing
    with pytest.raises(archive.ArchiveUnavailableError, match="отсутствуют погодные"):
        archive.fetch_issue_weather(context["one"], ISSUE)


@pytest.mark.parametrize("etag", ['"0123456789abcdef0123456789abcdef-50"', '"unknown"'])
def test_multipart_last_modified_is_not_treated_as_completed_publication(etag):
    payload = listing(RUN).replace(b'"0123456789abcdef0123456789abcdef"', etag.encode())
    with pytest.raises(archive.ArchiveUnavailableError, match="Multipart"):
        archive.parse_publication_listing(payload, RUN)


def test_malformed_provider_xml_becomes_archive_error():
    with pytest.raises(archive.ArchiveUnavailableError, match="XML"):
        archive.parse_publication_listing(b"not XML", RUN)


def test_missing_publication_field_falls_back_without_false_verification(context):
    def missing(run):
        payload = listing(run)
        if run == RUN:
            payload = payload.replace(b"<LastModified>", b"<Unknown>").replace(
                b"</LastModified>", b"</Unknown>"
            )
        return payload

    context["listing"] = missing
    result = archive.fetch_issue_weather(context["one"], ISSUE)
    assert result["run"] == (RUN - timedelta(days=1)).isoformat()


def test_offline_loader_excludes_analysis_and_preserves_per_lead_availability(context):
    context["listing"] = lambda run: listing(run, changed={14: run + timedelta(hours=5)})
    archive.fetch_issue_weather(context["one"], ISSUE)
    frame = archive.load_archive_frame(context["one"])
    assert len(frame) == 96
    assert frame.time.gt(frame.run).all()
    assert not frame.duplicated(["run", "time"]).any()
    assert frame.available_at.nunique() == 2
    assert frame.attrs["provenance"]["archive_hashes"][0]["point_sha256"]
    assert len(context["requests"]) == 1


def test_corrupt_blob_is_never_silently_used(context):
    result = archive.fetch_issue_weather(context["one"], ISSUE)
    path = archive._root() / "blobs" / f"{result['sha256']}.json"
    path.write_text("{}")
    with pytest.raises(archive.ArchiveUnavailableError, match="Контрольная сумма"):
        archive.fetch_issue_weather(context["one"], ISSUE, offline=True)


def test_loader_rejects_duplicate_run_manifest(context):
    archive.fetch_issue_weather(context["one"], ISSUE)
    path = next((archive._root() / "publication").glob("*.json"))
    (path.parent / "duplicate.json").write_bytes(path.read_bytes())
    with pytest.raises(archive.ArchiveUnavailableError, match="Дублирующийся"):
        archive.load_archive_frame(context["one"])


def test_forecast_is_invariant_to_mutations_outside_selected_horizon(context):
    first = archive.fetch_issue_weather(context["one"], ISSUE)
    # New extraction under a distinct cache, same requested inputs and a mutated unused future.
    for path in (archive._root() / "points").glob("*.json"):
        path.unlink()

    def mutate(run, latitude, longitude):
        data = raw_point(run, latitude, longitude)
        for point in data["points"][61:]:
            point["wind_u_100m"] = 5000
        return data

    context["point"] = mutate
    second = archive.fetch_issue_weather(context["one"], ISSUE)
    assert first["points"] == second["points"]
    assert first["available_at"] == second["available_at"]


def test_batch_deduplicates_same_grid_and_reports_failed_daily_runs(context):
    result = archive.download_archive([context["one"], context["two"]], ISSUE, ISSUE, workers=1)
    assert result == {"completed": 1, "failed": [], "tasks": 1, "unique_grid_cells": 1}
    assert len(context["points"]) == 1


def test_batch_bounds_are_enforced_before_network(context):
    with pytest.raises(ValueError, match="Лимит"):
        archive.download_archive([context["one"]], "2024-01-01T12:00Z", ISSUE)
    assert not context["requests"]


def test_reader_exact_run_selection_never_uses_nearest_future_cycle(monkeypatch):
    run = RUN.replace(tzinfo=None)
    coords = {
        "init_time": [run],
        "lead_time": np.arange(97).astype("timedelta64[h]"),
        "latitude": [43.75],
        "longitude": [78.5],
    }
    ds = xr.Dataset(
        {
            key: (("init_time", "lead_time", "latitude", "longitude"), np.ones((1, 97, 1, 1)))
            for key in archive.RAW_VARIABLES
        },
        coords=coords,
    )
    for key in archive.RAW_VARIABLES:
        ds[key].attrs["units"] = "degree_Celsius" if key == "temperature_2m" else "m s-1"
    monkeypatch.setattr(archive, "_dataset", lambda: ds)
    with pytest.raises(archive.ArchiveUnavailableError, match="Выпуск/ячейка"):
        archive._download_point(RUN - timedelta(days=1), 43.75, 78.5)
    result = archive._download_point(RUN, 43.75, 78.5)
    assert result["run"] == RUN.isoformat()
    ds["wind_u_100m"].attrs["units"] = "km/h"
    with pytest.raises(archive.ArchiveUnavailableError, match="единицы"):
        archive._download_point(RUN, 43.75, 78.5)


def test_cache_manifest_cannot_escape_archive_blob_directory(context):
    archive.fetch_issue_weather(context["one"], ISSUE)
    path = next((archive._root() / "publication").glob("*.json"))
    meta = json.loads(path.read_bytes())
    meta["raw_path"] = "../../outside.json"
    path.write_text(json.dumps(meta))
    with pytest.raises(archive.ArchiveUnavailableError, match="Недопустимый путь"):
        archive.fetch_issue_weather(context["one"], ISSUE, offline=True)
