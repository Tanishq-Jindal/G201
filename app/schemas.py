from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class PredictionInput(BaseModel):

    model_config = ConfigDict(extra="forbid")

    nearby_shops: int = Field(ge=0)

    days_to_salary: int = Field(ge=0)

    footfall_index: float = Field(ge=0)

    distance_to_market_km: float = Field(ge=0)

    avg_ticket_rs: float = Field(gt=0)

    other_atms_1km: int = Field(ge=0, le=25)

    atm_type: Literal[
        "bank_branch",
        "mall",
        "standalone"
    ]

    area: Literal[
        "Market",
        "Residential",
        "Highway"
    ]


class PredictionOutput(BaseModel):

    prediction: float