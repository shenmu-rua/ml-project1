"""Pydantic contracts for the collision triage API.

Field names and bounds mirror ``schema.json`` so a request can be fed straight
into the shared feature pipeline. All participants are optional: a caller that
only knows time, place and road context still gets a prediction, and the model
treats the missing groups the way it was trained to.
"""

from datetime import datetime
from typing import Annotated, Optional, Union

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

SPEED_LIMITS = (20, 30, 40, 50, 60, 70)


def _as_text(value: object) -> object:
    """DfT codes arrive as numbers from a form but the encoder expects text."""
    return None if value is None else str(value)


Category = Annotated[str, BeforeValidator(_as_text)]
Numeric = Optional[float]


class VehicleInput(BaseModel):
    """One vehicle row; only the columns listed in schema.json are accepted."""

    model_config = ConfigDict(extra="forbid")

    vehicle_type: Category | None = None
    towing_and_articulation: Category | None = None
    propulsion_code: Category | None = None
    sex_of_driver: Category | None = None
    journey_purpose_of_driver: Category | None = None
    escooter_flag: Category | None = None
    vehicle_left_hand_drive: Category | None = None
    vehicle_manoeuvre: Category | None = None
    junction_location: Category | None = None
    first_point_of_impact: Category | None = None
    skidding_and_overturning: Category | None = None
    hit_object_in_carriageway: Category | None = None
    vehicle_leaving_carriageway: Category | None = None
    hit_object_off_carriageway: Category | None = None
    vehicle_location_restricted_lane: Category | None = None
    age_of_driver: Numeric = None
    engine_capacity_cc: Numeric = None
    driver_imd_decile: Numeric = None


class CasualtyInput(BaseModel):
    """One casualty row; injury outcome codes are deliberately not accepted."""

    model_config = ConfigDict(extra="forbid")

    casualty_class: Category | None = None
    casualty_type: Category | None = None
    sex_of_casualty: Category | None = None
    pedestrian_location: Category | None = None
    pedestrian_movement: Category | None = None
    car_passenger: Category | None = None
    bus_or_coach_passenger: Category | None = None
    age_of_casualty: Numeric = None
    casualty_imd_decile: Numeric = None


class PredictRequest(BaseModel):
    """A single collision as known at dispatch time."""

    model_config = ConfigDict(extra="forbid")

    collision_datetime_utc: datetime = Field(
        ..., description="UTC timestamp of the collision"
    )
    latitude: float = Field(..., ge=49, le=62, description="WGS84 latitude")
    longitude: float = Field(..., ge=-9, le=3, description="WGS84 longitude")

    speed_limit: Numeric = Field(None, description="mph: 20/30/40/50/60/70")
    number_of_vehicles: Numeric = Field(None, ge=1, description="count")
    temp: Numeric = Field(None, ge=-100, le=70, description="degC")
    rhum: Numeric = Field(None, ge=0, le=100, description="percent")
    prcp: Numeric = Field(None, ge=0, description="mm in one hour")
    wdir: Numeric = Field(None, ge=0, le=360, description="degrees")
    wspd: Numeric = Field(None, ge=0, description="km/h")
    pres: Numeric = Field(None, ge=500, le=1100, description="hPa at sea level")

    road_type: Category | None = None
    junction_detail: Category | None = None
    junction_control: Category | None = None
    pedestrian_crossing: Category | None = None
    weather_conditions: Category | None = None
    road_surface_conditions: Category | None = None
    urban_or_rural_area: Category | None = None
    first_road_class: Category | None = None
    second_road_class: Category | None = None
    light_conditions: Category | None = None

    vehicles: list[VehicleInput] = Field(
        default_factory=list, description="on-scene vehicle list, optional"
    )
    casualties: list[CasualtyInput] = Field(
        default_factory=list, description="on-scene casualty list, optional"
    )

    @field_validator("speed_limit")
    @classmethod
    def _known_speed_limit(cls, value: Union[float, None]) -> Union[float, None]:
        if value is not None and value not in SPEED_LIMITS:
            raise ValueError(f"speed_limit must be one of {SPEED_LIMITS}")
        return value


class FeatureContribution(BaseModel):
    """One raw field's push on the log-odds; positive means "more severe"."""

    feature: str
    value: str
    impact: float


class UsageStats(BaseModel):
    """Cheap observability numbers D can log without touching the payload."""

    input_fields: int
    participant_rows: int
    explained_features: int
    elapsed_ms: int


class PredictResult(BaseModel):
    """What the service returns before the router adds the trace id."""

    probability: float = Field(..., ge=0, le=1, description="P(severe)")
    priority: int = Field(..., ge=1, le=3, description="1 immediate, 3 routine")
    priority_label: str
    base_value: float = Field(..., description="log-odds before contributions")
    contributions: list[FeatureContribution]
    other_impact: float
    cutoffs: list[float] = Field(..., description="[routine/urgent, urgent/immediate]")
    model_uri: str
    usage: UsageStats


class PredictResponse(PredictResult):
    """Response mirroring the contract: always carries a trace id."""

    trace_id: str
