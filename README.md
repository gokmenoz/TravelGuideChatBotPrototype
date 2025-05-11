# Travel Guide Chatbot

A travel guide chatbot that uses RAG (Retrieval-Augmented Generation) and Claude to provide travel information. The chatbot can answer questions about destinations, provide weather information, and visa requirements.

## Features

- 🤖 Powered by Claude 3 Sonnet
- 🔍 RAG-based knowledge retrieval
- 🌤️ Real-time weather information
- 📝 Visa requirement information
- 💬 Streaming responses
- 📚 Dynamic knowledge base expansion

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/travel-guide-chatbot.git
cd travel-guide-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

4. Set up environment variables:
```bash
# Create a .env file with:
OPENWEATHER_API_KEY="your_openweather_api_key"
```

5. Configure AWS credentials for Claude access:
```bash
aws configure --profile ogokmen_bedrock
```

## Usage

1. Start the API server:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8001 --reload
```

2. Make a request:
```bash
curl -X POST http://127.0.0.1:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What can I do for 3 days in Lisbon?"}'
```

## API Endpoints

- `POST /chat`: Main chat endpoint
  - Request body: `{"message": "your question", "history": []}`
  - Response: `{"response": "answer", "location": "detected_location"}`

- `GET /weather?city={city}`: Get weather information for a city
- `GET /visa?country={country}`: Get visa information for a country
- `GET /health`: Health check endpoint
- `GET /`: Root endpoint with status message

## Project Structure

```
travel-guide-chatbot/
├── src/
│   ├── api.py          # FastAPI application
│   ├── utils.py        # Utility functions
│   └── constants.py    # Constants and configuration
├── faiss_index/        # FAISS index and chunks (gitignored)
├── travel_docs/        # Cached travel documents (gitignored)
├── setup.py           # Package configuration
└── README.md          # This file
```

## Dependencies

- FastAPI: Web framework
- FAISS: Vector similarity search
- Sentence Transformers: Text embeddings
- Claude: Language model
- OpenWeather API: Weather information
- REST Countries API: Country information

## Development

1. Build the FAISS index:
```bash
python src/utils.py
```

2. Run tests (when available):
```bash
pytest
```

## License

MIT License

## API Usage

Start the server:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8001 --reload
```

Example API call:
```bash
curl -X POST http://127.0.0.1:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What can I do for 3 days in Lisbon?"}'
```
