from typing import Dict, Literal

from pydantic import BaseModel, Field


class PlayerInput(BaseModel):
    """Input payload for a single player engagement prediction."""

    SessionsPerWeek: int = Field(
        ...,
        ge=0,
        json_schema_extra={"example": 5},
    )
    AvgSessionDurationMinutes: float = Field(
        ...,
        ge=0,
        json_schema_extra={"example": 45.0},
    )
    PlayerLevel: int = Field(
        ...,
        ge=0,
        json_schema_extra={"example": 12},
    )
    AchievementsUnlocked: int = Field(
        ...,
        ge=0,
        json_schema_extra={"example": 8},
    )
    InGamePurchases: int = Field(
        ...,
        ge=0,
        le=1,
        json_schema_extra={"example": 1},
    )
    Gender: str = Field(
        ...,
        min_length=1,
        json_schema_extra={"example": "Male"},
    )
    Location: str = Field(
        ...,
        min_length=1,
        json_schema_extra={"example": "Nigeria"},
    )
    GameGenre: str = Field(
        ...,
        min_length=1,
        json_schema_extra={"example": "Strategy"},
    )

    class Config:
        json_schema_extra = {
            "example": {
                "SessionsPerWeek": 5,
                "AvgSessionDurationMinutes": 45.0,
                "PlayerLevel": 12,
                "AchievementsUnlocked": 8,
                "InGamePurchases": 1,
                "Gender": "Male",
                "Location": "Nigeria",
                "GameGenre": "Strategy",
            }
        }


class PredictionOutput(BaseModel):
    """Prediction response returned by the engagement model."""

    predicted_class: Literal["Low", "Medium", "High"] = Field(
        ...,
        json_schema_extra={"example": "Medium"},
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        json_schema_extra={"example": 0.8734},
    )
    all_probabilities: Dict[str, float] = Field(
        ...,
        json_schema_extra={
            "example": {
                "Low": 0.0312,
                "Medium": 0.8734,
                "High": 0.0954,
            }
        },
    )
