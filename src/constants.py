import os
from dotenv import load_dotenv

# Load variables from .env file into the environment
load_dotenv()

# Now safely read the API key
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Optional: Raise an error if the key is missing
if not OPENWEATHER_API_KEY:
    raise RuntimeError("❌ OPENWEATHER_API_KEY is not set in your .env file.")

QUESTIONS = [
    "What are the best neighborhoods to stay in Berlin?",
    "What can I see in Barcelona in one day?",
    "Is Istanbul safe for tourists?",
    "How do I travel from Rome to Florence?",
    "What are kid-friendly activities in New York?",
]