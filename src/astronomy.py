"""Astral calculations at collision coordinates; these are not weather readings."""

from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd
from astral import Observer
from astral.sun import dawn, dusk, elevation, sunrise, sunset

EVENTS = (
    "sunrise_utc",
    "sunset_utc",
    "civil_dawn_utc",
    "civil_dusk_utc",
    "nautical_dawn_utc",
    "nautical_dusk_utc",
    "astronomical_dawn_utc",
    "astronomical_dusk_utc",
)


def solar_phase(angle):
    """Geometric solar altitude; sunrise/sunset use the standard -0.833° horizon."""
    return np.select(
        [angle >= -0.833, angle >= -6, angle >= -12, angle >= -18],
        ["day", "civil_twilight", "nautical_twilight", "astronomical_twilight"],
        default="night",
    )


def solar_angles(frame):
    times = pd.to_datetime(frame["collision_datetime_utc"], utc=True, errors="raise")
    if times.isna().any():
        raise ValueError("Solar features need an unambiguous UTC collision time.")
    values = [
        elevation(
            Observer(float(lat), float(lon)),
            moment.to_pydatetime(),
            with_refraction=False,
        )
        for lat, lon, moment in zip(frame.latitude, frame.longitude, times)
    ]
    return np.asarray(values, dtype=float)


@lru_cache(maxsize=250_000)
def _events(latitude, longitude, local_date):
    observer = Observer(latitude, longitude, elevation=0)
    day = date.fromisoformat(local_date)
    calls = (
        lambda: sunrise(observer, day, tzinfo="UTC"),
        lambda: sunset(observer, day, tzinfo="UTC"),
        lambda: dawn(observer, day, depression=6, tzinfo="UTC"),
        lambda: dusk(observer, day, depression=6, tzinfo="UTC"),
        lambda: dawn(observer, day, depression=12, tzinfo="UTC"),
        lambda: dusk(observer, day, depression=12, tzinfo="UTC"),
        lambda: dawn(observer, day, depression=18, tzinfo="UTC"),
        lambda: dusk(observer, day, depression=18, tzinfo="UTC"),
    )
    result = []
    for call in calls:
        try:
            result.append(call().isoformat())
        except ValueError:
            # In northern UK summers a twilight threshold may never be crossed.
            result.append(None)
    return tuple(result)


def add_astronomy(frame, progress=False):
    """Add rise/set and all three twilight levels without filling absent events."""
    result = frame.copy()
    times = pd.to_datetime(frame.collision_datetime_utc, utc=True)
    local_dates = times.dt.tz_convert("Europe/London").dt.strftime("%Y-%m-%d")
    valid = times.notna() & frame.latitude.notna() & frame.longitude.notna()
    records = [None] * len(frame)
    for count, position in enumerate(np.flatnonzero(valid.to_numpy()), 1):
        records[position] = _events(
            float(frame.latitude.iloc[position]),
            float(frame.longitude.iloc[position]),
            local_dates.iloc[position],
        )
        if progress and count % 25_000 == 0:
            print(f"Astronomy calculated: {count:,}", flush=True)
    events = pd.DataFrame(
        [r or (None,) * len(EVENTS) for r in records], columns=EVENTS, index=frame.index
    )
    for column in EVENTS:
        result[column] = pd.to_datetime(events[column], utc=True)
    result["sun_elevation_deg"] = np.nan
    result.loc[valid, "sun_elevation_deg"] = solar_angles(frame.loc[valid])
    result["solar_phase"] = pd.Series(
        solar_phase(result.sun_elevation_deg.to_numpy()), index=frame.index
    )
    result.loc[~valid, "solar_phase"] = None
    result["daylight_minutes"] = (
        result.sunset_utc - result.sunrise_utc
    ).dt.total_seconds() / 60
    result["minutes_after_sunrise"] = (
        times - result.sunrise_utc
    ).dt.total_seconds() / 60
    result["minutes_to_sunset"] = (result.sunset_utc - times).dt.total_seconds() / 60
    result["astronomical_twilight_absent"] = valid & result.astronomical_dawn_utc.isna()
    return result
