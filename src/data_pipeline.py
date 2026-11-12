"""Reproducible Python ingestion, observed-weather linkage and temporal splits.

Run from the repository root: ``python -m src.data_pipeline all``.
Raw CSVs in data/raw/ are read-only; all generated data go into processed/ and splits/.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from geographiclib.geodesic import Geodesic

from src.astronomy import add_astronomy
from src.participants import RAW_DIR, feature_names, read_participants, table_paths
from src.validation import load_schema, validate_frame

ROOT = Path(__file__).resolve().parents[1]
CORE = ["temp", "rhum", "prcp", "wdir", "wspd", "pres"]
NS_MINUTE = 60_000_000_000


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def time_ns(series):
    return pd.DatetimeIndex(series).to_numpy(dtype="datetime64[ns]").astype(np.int64)


def valid_range(values, definition):
    good = np.isfinite(values)
    if "min" in definition:
        good &= values >= definition["min"]
    if "max" in definition:
        good &= values <= definition["max"]
    if "allowed" in definition:
        good &= np.isin(values, definition["allowed"])
    return good


def weather_path(data_dir):
    return Path(data_dir) / RAW_DIR / "weather_2024_2025.csv"


def read_collisions(data_dir, contract):
    source = pd.concat(
        [
            pd.read_csv(path, dtype=str, keep_default_na=False)
            for path in table_paths(data_dir, "collision")
        ],
        ignore_index=True,
    )
    original_rows = len(source)
    source = source.drop_duplicates().reset_index(drop=True)
    if source.collision_index.duplicated().any():
        raise ValueError(
            "Conflicting records share a collision_index; inspect the raw files."
        )
    result = pd.DataFrame({"collision_index": source.collision_index.astype(object)})
    result["collision_severity"] = pd.to_numeric(
        source.collision_severity, errors="coerce"
    )
    result["is_severe"] = source.collision_severity.map(
        contract["target"]["mapping"]
    ).astype(float)
    naive = pd.to_datetime(
        source.date + " " + source.time, format="%d/%m/%Y %H:%M", errors="coerce"
    )
    localized = naive.dt.tz_localize(
        "Europe/London", ambiguous="NaT", nonexistent="NaT"
    )
    alternate = naive.dt.tz_localize(
        "Europe/London", ambiguous=False, nonexistent="NaT"
    )
    result["collision_datetime_utc"] = localized.dt.tz_convert("UTC")
    result["collision_date_local"] = naive.dt.strftime("%Y-%m-%d")
    result["collision_year"] = pd.to_numeric(source.collision_year).astype(int)
    result["time_issue"] = np.select(
        [
            naive.isna(),
            naive.notna() & localized.isna() & alternate.notna(),
            naive.notna() & localized.isna() & alternate.isna(),
        ],
        ["invalid_date_or_time", "ambiguous_DST_hour", "nonexistent_DST_hour"],
        default="",
    )
    invalid_counts = {}
    for name, definition in contract["numeric_features"].items():
        if name in CORE:
            continue
        values = pd.to_numeric(source[name], errors="coerce").to_numpy(
            dtype=float, copy=True
        )
        valid = valid_range(values, definition)
        invalid_counts[name] = int((~valid).sum())
        result[name] = np.where(valid, values, np.nan)
    for name, definition in contract["categorical_features"].items():
        values = source[name].astype(object).copy()
        # Code 0 explicitly means this control is not applicable away from junctions.
        if name == "junction_control":
            values.loc[source.junction_detail.eq("0")] = "0"
        valid = values.isin(definition["categories"])
        invalid_counts[name] = int((~valid).sum())
        result[name] = values.where(valid, np.nan).astype(object)
    report = {
        "raw_rows": original_rows,
        "exact_duplicate_rows_removed": original_rows - len(source),
        "conflicting_duplicate_ids": 0,
        "rows_retained_in_master": len(result),
        "original_severity_counts": source.collision_severity.value_counts().to_dict(),
        "invalid_or_unknown_cells_set_missing": invalid_counts,
        "time_issues": result.time_issue.value_counts().to_dict(),
    }
    return result, report


def clean_weather(path, contract):
    """Mask unqualified values, never substitute a forecast/grid estimate."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if raw.duplicated(["native_file_id", "native_row"]).any():
        raise ValueError("Duplicate native weather IDs require source review.")
    weather = raw[["provider", "station_id", "native_file_id", "native_row"]].copy()
    weather["weather_id"] = raw.native_file_id + ":" + raw.native_row
    weather["datetime_utc"] = pd.to_datetime(raw.time_utc, utc=True)
    for field in ("latitude", "longitude"):
        weather[field] = pd.to_numeric(raw[field], errors="coerce")
    providers = raw.provider
    masks = {
        p: providers.eq(p).to_numpy() for p in ("Meteostat", "NOAA_GHCNh", "COSMOS_UK")
    }
    values = {
        p: pd.to_numeric(raw[p], errors="coerce").to_numpy(dtype=float, copy=True)
        for p in CORE
    }
    values["wspd"] *= np.where(
        raw.wspd_unit.eq("m/s"), 3.6, np.where(raw.wspd_unit.eq("km/h"), 1.0, np.nan)
    )
    accepted, checked, derived = {}, {}, {}
    for field in CORE:
        bounds = valid_range(values[field], contract["weather"]["bounds"][field])
        flag = raw[field + "_flag"].str.strip()
        source = raw[field + "_source"].str.strip()
        met = (
            masks["Meteostat"]
            & source.isin(contract["weather"]["meteostat_observed_sources"]).to_numpy()
            & bounds
        )
        noaa = (
            masks["NOAA_GHCNh"]
            & flag.eq("1").to_numpy()
            & source.ne("").to_numpy()
            & bounds
        )
        is_derived = np.zeros(len(raw), dtype=bool)
        if field == "rhum":
            dewpoint = pd.to_numeric(raw.dewpoint, errors="coerce").to_numpy(
                dtype=float
            )
            is_derived = (
                masks["NOAA_GHCNh"] & raw.rhum_measurement_code.eq("D").to_numpy()
            )
            is_derived &= (
                raw.temp_flag.eq("1").to_numpy() & raw.dewpoint_flag.eq("1").to_numpy()
            )
            is_derived &= np.isfinite(values["temp"]) & np.isfinite(dewpoint) & bounds
            is_derived &= flag.isin(["", "1"]).to_numpy() & source.ne("").to_numpy()
            noaa |= is_derived
        if field == "prcp":
            noaa &= (
                raw.prcp_measurement_code.eq("").to_numpy()
                & raw.prcp_report_type.isin(["FM12", "FM13", "FM14"]).to_numpy()
            )
        cosmos = masks["COSMOS_UK"] & flag.eq("").to_numpy() & bounds
        if field in ("pres", "prcp"):
            cosmos[:] = False
        accepted[field] = met | noaa | cosmos
        checked[field] = noaa | cosmos
        derived[field] = is_derived
    # Native COSMOS precipitation is 30 minutes. Keep both inputs from one gauge.
    cosmos = raw.loc[providers.eq("COSMOS_UK")].copy()
    cosmos["_time"] = pd.to_datetime(cosmos.time_utc, utc=True)
    cosmos = cosmos.sort_values(["station_id", "_time"], kind="stable")
    previous_time = cosmos.groupby("station_id", sort=False)["_time"].shift()
    consecutive = (cosmos._time - previous_time).eq(pd.Timedelta(minutes=30)).to_numpy()
    rain = np.full(len(cosmos), np.nan)
    rain_source = np.full(len(cosmos), "", dtype=object)
    for variable, flag_column in (
        ("prcp", "prcp_flag"),
        ("precip_raine", "precip_raine_flag"),
        ("precip_tipping", "precip_tipping_flag"),
    ):
        amount = pd.to_numeric(cosmos[variable], errors="coerce")
        good = amount.notna() & amount.ge(0) & cosmos[flag_column].eq("")
        previous_good = good.groupby(cosmos.station_id, sort=False).shift(
            fill_value=False
        )
        previous_amount = amount.groupby(cosmos.station_id, sort=False).shift()
        choose = (
            good.to_numpy()
            & previous_good.to_numpy()
            & consecutive
            & ~np.isfinite(rain)
        )
        rain[choose] = (amount + previous_amount).to_numpy()[choose]
        rain_source[choose] = variable
    indices = cosmos.index.to_numpy()
    values["prcp"][indices] = rain
    valid_rain = np.isfinite(rain)
    accepted["prcp"][indices] = valid_rain
    checked["prcp"][indices] = valid_rain
    derived["prcp"][indices] = valid_rain
    weather["rain_input_variable"] = ""
    weather.loc[indices, "rain_input_variable"] = rain_source
    for field in CORE:
        weather[field] = np.where(accepted[field], values[field], np.nan)
        weather[field + "_qc_verified"] = checked[field]
        weather[field + "_derived"] = derived[field]
    weather["n_present"] = weather[CORE].notna().sum(axis=1).astype(int)
    weather["n_checked"] = np.column_stack(
        [checked[p] & accepted[p] for p in CORE]
    ).sum(axis=1)
    weather["n_derived"] = np.column_stack(
        [derived[p] & accepted[p] for p in CORE]
    ).sum(axis=1)
    report = {
        "input_rows": len(raw),
        "output_rows": len(weather),
        "rows_with_any_observed_field": int(weather.n_present.gt(0).sum()),
        "missing_after_source_and_qc_checks": {
            p: int(weather[p].isna().sum()) for p in CORE
        },
        "source_rows": raw.provider.value_counts().to_dict(),
        "self_imputed_values": 0,
        "station_pressure_not_used_as_sea_level_pressure": True,
    }
    return weather, report


def select_in_window(times, targets, present, checked, derived, tolerance):
    """Quality-first rule, then nearest time, with earlier observation on ties."""
    selected = np.full(len(targets), -1, dtype=np.int64)
    priority = np.where(present > 0, (present * 7 + checked) * 7 + (6 - derived), -1)
    for begin in range(0, len(targets), 4096):
        stop = min(begin + 4096, len(targets))
        query = targets[begin:stop]
        left = np.searchsorted(times, query - tolerance, side="left")
        right = np.searchsorted(times, query + tolerance, side="right")
        width = int(np.max(right - left, initial=0))
        if width == 0:
            continue
        positions = left[:, None] + np.arange(width)[None, :]
        valid = positions < right[:, None]
        clipped = np.minimum(positions, len(times) - 1)
        ranks = np.where(valid, priority[clipped], -1)
        maximum = ranks.max(axis=1)
        eligible = valid & (ranks == maximum[:, None]) & (maximum[:, None] >= 0)
        differences = np.where(
            eligible, np.abs(times[clipped] - query[:, None]), np.iinfo(np.int64).max
        )
        column = differences.argmin(axis=1)
        chosen = positions[np.arange(len(query)), column]
        selected[begin:stop] = np.where(maximum >= 0, chosen, -1)
    return selected


def haversine(lat, lon, station_lat, station_lon):
    angle = np.sin(np.radians(lat - station_lat) / 2) ** 2
    angle += (
        np.cos(np.radians(lat))
        * np.cos(np.radians(station_lat))
        * np.sin(np.radians(lon - station_lon) / 2) ** 2
    )
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(angle, 0, 1)))


def match_weather(collisions, weather, contract, progress=True):
    max_km = contract["weather"]["maximum_distance_km"]
    tolerance = int(contract["weather"]["maximum_time_difference_minutes"] * NS_MINUTE)
    n = len(collisions)
    lat = collisions.latitude.to_numpy(dtype=float)
    lon = collisions.longitude.to_numpy(dtype=float)
    valid = (
        collisions.collision_datetime_utc.notna().to_numpy()
        & np.isfinite(lat)
        & np.isfinite(lon)
    )
    targets = time_ns(collisions.collision_datetime_utc)
    chosen = np.full(n, -1, dtype=np.int64)
    distances = np.full(n, np.nan)
    offsets = np.full(n, np.nan)
    scores = [None] * n
    cache = {}
    groups = weather.groupby(
        ["provider", "station_id", "latitude", "longitude"], sort=True, dropna=True
    )
    for group_number, (
        (provider, station, station_lat, station_lon),
        group,
    ) in enumerate(groups, 1):
        location = (float(station_lat), float(station_lon))
        if location not in cache:
            approximate = haversine(lat, lon, station_lat, station_lon)
            candidates = np.flatnonzero(valid & (approximate <= max_km * 1.01))
            nearby = []
            for i in candidates:
                distance = (
                    Geodesic.WGS84.Inverse(lat[i], lon[i], station_lat, station_lon)[
                        "s12"
                    ]
                    / 1000
                )
                if distance <= max_km:
                    nearby.append((i, distance))
            cache[location] = nearby
        nearby = cache[location]
        if not nearby:
            continue
        group = group.sort_values(
            ["datetime_utc", "native_file_id", "native_row"], kind="stable"
        )
        times = time_ns(group.datetime_utc)
        indices = np.asarray([i for i, _ in nearby], dtype=np.int64)
        present = group.n_present.to_numpy()
        checked = group.n_checked.to_numpy()
        derived = group.n_derived.to_numpy()
        picked = select_in_window(
            times, targets[indices], present, checked, derived, tolerance
        )
        source_order = "Meteostat_observed" if provider == "Meteostat" else provider
        source_positions = group.index.to_numpy()
        for (i, distance), k in zip(nearby, picked):
            if k < 0:
                continue
            score = (
                -int(present[k]),
                -int(checked[k]),
                int(derived[k]),
                distance,
                abs(int(times[k] - targets[i])),
                int(times[k]),
                source_order,
                station,
            )
            if scores[i] is None or score < scores[i]:
                scores[i] = score
                chosen[i] = source_positions[k]
                distances[i] = distance
                offsets[i] = (int(times[k]) - int(targets[i])) / NS_MINUTE
        if progress and group_number % 50 == 0:
            print(
                f"Weather station/coordinate groups checked: {group_number}/{len(groups)}",
                flush=True,
            )
    result = collisions.copy()
    good = chosen >= 0
    selected = weather.reindex(np.where(good, chosen, -1)).reset_index(drop=True)
    for field in CORE:
        result[field] = selected[field].to_numpy(dtype=float)
    for field, name in (
        ("weather_id", "weather_id"),
        ("provider", "weather_provider"),
        ("station_id", "weather_station_id"),
        ("datetime_utc", "weather_datetime_utc"),
        ("n_present", "weather_present_count"),
        ("n_checked", "weather_qc_count"),
        ("n_derived", "weather_derived_count"),
    ):
        result[name] = selected[field].to_numpy()
    result["weather_distance_km"] = distances
    result["weather_time_offset_minutes"] = offsets
    complete = result.weather_present_count.eq(6)
    qc_complete = complete & result.weather_qc_count.eq(6)
    result["weather_quality_tier"] = np.select(
        [qc_complete, complete, good],
        ["qc_core6", "observed_core6", "partial_observed"],
        default="no_usable_match",
    )
    result["model_eligible"] = qc_complete & result.is_severe.notna() & valid
    result["exclusion_reason"] = np.select(
        [
            result.is_severe.isna(),
            result.latitude.isna() | result.longitude.isna(),
            result.time_issue.ne(""),
            ~complete,
            ~qc_complete,
        ],
        [
            "invalid_target",
            "invalid_coordinates",
            "unresolved_local_time",
            "core_weather_incomplete",
            "weather_qc_not_exposed",
        ],
        default="",
    )
    report = {
        "accidents": n,
        "any_observed_weather": int(good.sum()),
        "observed_core6_complete": int(complete.sum()),
        "core6_all_qc_checkable": int(qc_complete.sum()),
        "model_eligible": int(result.model_eligible.sum()),
        "exclusion_reasons": result.exclusion_reason.value_counts().to_dict(),
        "selected_provider_counts": result.loc[good, "weather_provider"]
        .value_counts()
        .to_dict(),
        "maximum_distance_km": float(np.nanmax(distances)),
        "maximum_absolute_time_offset_minutes": float(np.nanmax(np.abs(offsets))),
    }
    return result, report


def prepare(data_dir=ROOT / "data"):
    data_dir = Path(data_dir)
    contract = load_schema()
    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    paths = table_paths(data_dir, "collision") + [weather_path(data_dir)]
    participant_paths = [
        path
        for table in contract["participants"]
        for path in table_paths(data_dir, table)
    ]
    hashes = {path.name: sha256(path) for path in paths + participant_paths}
    collisions, collision_report = read_collisions(data_dir, contract)
    weather, weather_report = clean_weather(paths[-1], contract)
    print("Raw CSVs read; observed weather values validated.", flush=True)
    master, match_report = match_weather(collisions, weather, contract)
    if (
        hashes[paths[-1].name]
        == "419a3682b6c6f16adb210a3e343da339b3797f74822d1c3339c08ae6159a54bb"
        and len(collisions) == 202452
    ):
        expected = (202032, 180518, 164860)
        actual = tuple(
            match_report[key]
            for key in (
                "any_observed_weather",
                "observed_core6_complete",
                "core6_all_qc_checkable",
            )
        )
        if actual != expected:
            raise ValueError(
                f"Linkage differs from the audited source snapshot: {actual} != {expected}"
            )
    master = add_astronomy(master, progress=True)
    master = master.join(read_participants(data_dir, contract), on="collision_index")
    validate_frame(master)
    validate_frame(master.loc[master.model_eligible], modeling=True)
    weather.to_csv(
        processed / "weather_observations.csv", index=False, encoding="utf-8-sig"
    )
    master.to_csv(processed / "collisions_clean.csv", index=False, encoding="utf-8-sig")
    excluded = master.loc[
        ~master.model_eligible,
        ["collision_index", "exclusion_reason", "weather_quality_tier", "time_issue"],
    ]
    excluded.to_csv(
        processed / "excluded_from_model.csv", index=False, encoding="utf-8-sig"
    )
    report = {
        "schema_version": contract["schema_version"],
        "raw_sha256": hashes,
        "collisions": collision_report,
        "weather": weather_report,
        "matching": match_report,
        "raw_files_modified": False,
        "synthetic_weather_values_added": 0,
        "pandera_master_and_model_checks_passed": True,
        "astronomy": {
            "coordinates": "each collision",
            "library": "Astral 3.2",
            "observer_elevation_m": 0,
            "absent_astronomical_twilight_events": int(
                master.astronomical_twilight_absent.sum()
            ),
        },
    }
    write_json(data_dir / "quality_report.json", report)
    print("PREPARE_COMPLETE", json.dumps(match_report, ensure_ascii=False), flush=True)
    return master, report


def temporal_split(frame, settings):
    """Whole UTC dates, with a 2h separation between adjacent split populations."""
    times = pd.to_datetime(frame.collision_datetime_utc, utc=True)
    if times.isna().any():
        raise ValueError("Cannot split records with unresolved UTC timestamps.")
    counts = times.dt.floor("D").value_counts().sort_index()
    cumulative = counts.cumsum().to_numpy()
    first = int(np.searchsorted(cumulative, len(frame) * settings["train_fraction"]))
    second = int(
        np.searchsorted(
            cumulative,
            len(frame) * (settings["train_fraction"] + settings["validation_fraction"]),
        )
    )
    if first >= second or second >= len(counts) - 1:
        raise ValueError(
            "Too few distinct dates to construct three chronological splits."
        )
    boundary1 = counts.index[first] + pd.Timedelta(days=1)
    boundary2 = counts.index[second] + pd.Timedelta(days=1)
    gap = pd.Timedelta(minutes=settings["purge_minutes_each_side"])
    labels = pd.Series("purged", index=frame.index, dtype=object)
    labels.loc[times < boundary1 - gap] = "train"
    labels.loc[(times >= boundary1 + gap) & (times < boundary2 - gap)] = "val"
    labels.loc[times >= boundary2 + gap] = "test"
    return labels, [boundary1.isoformat(), boundary2.isoformat()]


def split(data_dir=ROOT / "data"):
    data_dir = Path(data_dir)
    contract = load_schema()
    from src.features import CATEGORICAL_INPUTS

    frame = pd.read_csv(
        data_dir / "processed/collisions_clean.csv",
        dtype={
            **{c: object for c in CATEGORICAL_INPUTS},
            "collision_index": str,
            "time_issue": str,
        },
        low_memory=False,
    )
    frame["collision_datetime_utc"] = pd.to_datetime(
        frame.collision_datetime_utc, utc=True
    )
    selected = frame.loc[
        frame.weather_quality_tier.eq("qc_core6") & frame.exclusion_reason.isna()
    ].copy()
    validate_frame(selected, modeling=True)
    labels, boundaries = temporal_split(selected, contract["split"])
    output = data_dir / "splits"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "method": contract["split"],
        "boundaries_utc": boundaries,
        "eligible_rows": len(selected),
        "splits": {},
    }
    keep = ["collision_index", "collision_datetime_utc", "collision_year"]
    keep += list(contract["numeric_features"]) + list(contract["categorical_features"])
    keep += feature_names(contract)
    keep += [
        "is_severe",
        "weather_id",
        "weather_provider",
        "weather_distance_km",
        "weather_time_offset_minutes",
    ]
    ids, weather_ids = {}, {}
    for name in ("train", "val", "test"):
        subset = selected.loc[labels.eq(name)].sort_values(
            ["collision_datetime_utc", "collision_index"]
        )
        validate_frame(subset, modeling=True)
        path = output / f"{name}.csv"
        subset[keep].to_csv(path, index=False, encoding="utf-8-sig")
        ids[name] = set(subset.collision_index)
        weather_ids[name] = set(subset.weather_id)
        report["splits"][name] = {
            "rows": len(subset),
            "positive_rows": int(subset.is_severe.sum()),
            "positive_rate": float(subset.is_severe.mean()),
            "start_utc": subset.collision_datetime_utc.min().isoformat(),
            "end_utc": subset.collision_datetime_utc.max().isoformat(),
            "sha256": sha256(path),
        }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if ids[left] & ids[right] or weather_ids[left] & weather_ids[right]:
            raise ValueError(
                f"Shared accident or weather IDs between {left} and {right}."
            )
    purged = selected.loc[
        labels.eq("purged"), ["collision_index", "collision_datetime_utc"]
    ]
    purged.to_csv(output / "boundary_gap.csv", index=False, encoding="utf-8-sig")
    report["boundary_gap_rows"] = len(purged)
    report["accident_ids_disjoint"] = report["weather_ids_disjoint"] = True
    write_json(data_dir / "split_report.json", report)
    print("SPLIT_COMPLETE", json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "split", "eda", "all"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    if args.stage in ("prepare", "all"):
        prepare(args.data_dir)
    if args.stage in ("split", "all"):
        split(args.data_dir)
    if args.stage in ("eda", "all"):
        from src.eda import generate_eda

        generate_eda(args.data_dir)


if __name__ == "__main__":
    main()
