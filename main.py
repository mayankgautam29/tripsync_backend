# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv
import os
import uvicorn

# Load .env file
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")
genai.configure(api_key=api_key)

app = FastAPI()

# CORS setup (allow frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# System prompt for model
SYSTEM_PROMPT = """
You are an AI travel planner that creates realistic and time-specific itineraries. 
Given location, dates, interests, and group size — produce a short, structured day-by-day plan. 
Do not include extra text, just give the answer simply.
"""

# Data model for request
class TripData(BaseModel):
    tripId: str
    location: str
    date_from: str
    date_to: str
    interests: list[str]
    group_size: dict


@app.post("/generate-itinerary")
def generate_itinerary(trip: TripData):
    try:
        # Build user prompt dynamically
        prompt_text = (
            f"Plan a detailed itinerary for a trip to {trip.location} "
            f"from {trip.date_from} to {trip.date_to}. "
            f"Group size: {trip.group_size.get('min', 2)}–{trip.group_size.get('max', 8)} people. "
            f"Interests: {', '.join(trip.interests)}.\n\n"
            f"Include timings, activities, and local recommendations."
        )

        model = genai.GenerativeModel(model_name="gemini-2.5-flash")

        response = model.generate_content([SYSTEM_PROMPT, prompt_text])
        itinerary_text = getattr(response, "text", None)
        if not itinerary_text:
            raise HTTPException(status_code=500, detail="Model returned no content")

        return {"itinerary": itinerary_text}

    except Exception as e:
        print(f"Error generating itinerary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "Trip Itinerary Generator API is running 🚀"}


# ✅ Run Uvicorn with dynamic port (for Render deployment)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render dynamically assigns a PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
