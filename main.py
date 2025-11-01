# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv
import os
import uvicorn

# Load environment variables from .env
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")
genai.configure(api_key=api_key)

# Initialize FastAPI app
app = FastAPI(title="AI Travel Itinerary API", version="1.0")

# Enable CORS (optional, for frontend usage)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request
class TripData(BaseModel):
    tripId: str = Field(..., description="Unique ID for the trip")
    location: str = Field(..., description="Destination of the trip")
    date_from: str = Field(..., description="Start date in YYYY-MM-DD format")
    date_to: str = Field(..., description="End date in YYYY-MM-DD format")
    interests: List[str] = Field(..., description="List of interests")
    group_size: Dict[str, int] = Field(..., description="Dictionary with min and max group size")

# Root endpoint
@app.get("/")
def home():
    return {"message": "Hello World! API is running 🚀"}

# Generate itinerary endpoint
@app.post("/generate-itinerary")
def generate_itinerary(trip: TripData):
    SYSTEM_PROMPT = """
    You are an AI travel planner that creates realistic and time-specific itineraries.
    Given location, dates, interests, and group size — produce a short, structured day-by-day plan.
    Only provide the itinerary, do not add extra explanations.
    """

    prompt_text = (
        f"Plan a detailed itinerary for a trip to {trip.location} "
        f"from {trip.date_from} to {trip.date_to}. "
        f"Group size: {trip.group_size.get('min', 2)}–{trip.group_size.get('max', 8)} people. "
        f"Interests: {', '.join(trip.interests)}.\n\n"
        f"Include timings, activities, and local recommendations."
    )

    try:
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        response = model.generate_content([SYSTEM_PROMPT, prompt_text])
        itinerary_text = getattr(response, "text", str(response))
        return {"tripId": trip.tripId, "itinerary": itinerary_text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate itinerary: {e}")

# Run server like Express.js
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))  # default port like Express
    print(f"🚀 Server is running on http://0.0.0.0:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
