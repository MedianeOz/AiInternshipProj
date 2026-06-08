from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.model import get_model_status, predict_engagement
from api.schemas import PlayerInput, PredictionOutput


app = FastAPI(title="BabiPoly Player Engagement API", version="1.0")

# Allow the future Streamlit frontend, Swagger UI, and local testing clients to
# call the API without browser CORS restrictions.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root() -> dict[str, str]:
    """Return a simple status message confirming that the API is running."""
    return {
        "status": "ok",
        "message": "BabiPoly Engagement API is running",
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(player: PlayerInput) -> PredictionOutput:
    """Predict the engagement level for a single player payload."""
    try:
        return predict_engagement(player)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc


@app.get("/health")
def health() -> dict:
    """Return model artifact and preprocessing load status."""
    status = get_model_status()
    status["status"] = "ok" if status["model_loaded"] else "error"
    return status
