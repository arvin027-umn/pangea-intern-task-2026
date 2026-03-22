import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.feedback import get_feedback
from app.models import FeedbackRequest, FeedbackResponse

"""FastAPI application for language feedback API
Provides two endpoints:
- GET /health: simple health check returning {"status": "ok"}
- POST /feedback: analyzes a learner's sentence and returns correction feedback
"""

load_dotenv()

# Logging setup to observe what's happening in the container
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Language Feedback API",
    description=(
        "Analyzes learner-written sentences and provides structured "
        "language correction feedback powered by an LLM."
    ),
    version="1.0.0",
)

@app.get("/health")
async def health() -> dict:
    """Health check endpoint, returns 200 if server is running"""
    return {"status": "ok"}

@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Analyze a learner's sentence and return correction feedback
    - Accepts a sentence in a language that the user is trying to learn
    along with the language user can converse in
    - Returns corrected sentence, a list of errors with explanations,
    whether the sentence was correct, and a CEFR difficulty rating
    """
    try:
        result = await get_feedback(request)
        return result
    except Exception as exc:
        # Logging error for debugging
        logger.exception("Error processing feedback request: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request. Please try again.",
        ) from exc