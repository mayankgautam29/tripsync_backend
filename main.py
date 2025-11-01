from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import os
import uvicorn

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")
genai.configure(api_key=api_key)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class TripData(BaseModel):
    tripId: str
    location: str
    date_from: str
    date_to: str
    interests: list[str]
    group_size: dict

# Root route (like Express `/`)
@app.get("/")
def home():
    return {"message": "Hello World! API is running 🚀"}

# AI itinerary route
@app.post("/generate-itinerary")
def generate_itinerary(trip: TripData):
    SYSTEM_PROMPT = """
    You are an AI travel planner that creates realistic and time-specific itineraries.
    Given location, dates, interests, and group size — produce a short, structured day-by-day plan.
    Do not include extra text, just give the answer simply.
    """

    prompt_text = (
        f"Plan a detailed itinerary for a trip to {trip.location} "
        f"from {trip.date_from} to {trip.date_to}. "
        f"Group size: {trip.group_size.get('min', 2)}–{trip.group_size.get('max', 8)} people. "
        f"Interests: {', '.join(trip.interests)}.\n\n"
        f"Include timings, activities, and local recommendations."
    )

    try:
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        response = model.generate_content([SYSTEM_PROMPT, prompt_text])
        return {"itinerary": response.text}
    except Exception as e:
        return {"error": str(e)}

# Run the app (like Express)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))  # Default like Express
    print(f"🚀 Server is running on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
