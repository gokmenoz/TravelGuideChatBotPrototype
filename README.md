# Travel Guide Chatbot Prototype

This repository contains the code for a Travel Guide Chatbot, built using Streamlit, Sentence Transformers, and vector search with FAISS. This chatbot assists users by answering travel-related questions using semantic search, retrieval-augmented generation (RAG), and generative AI models including Claude and LLaMA.

## Features

* **Semantic Search**: Utilizes sentence embeddings to match user queries with relevant travel information.
* **Interactive UI**: Simple and user-friendly interface powered by Streamlit.
* **Efficient Retrieval**: FAISS integration for rapid information retrieval.
* **RAG Algorithm**: Retrieval-augmented generation using a cached knowledge base for enhanced accuracy and efficiency.
* **Instruction Tuning**: Custom instruction tuning performed on LLaMA to improve chatbot responses.
* **Advanced Generative Models**: Uses Claude and LLaMA models to provide sophisticated and context-aware responses.

## Technologies Used

* Python 3.10
* Streamlit
* Sentence Transformers (`all-MiniLM-L6-v2`)
* PyTorch (optimized for Apple Silicon)
* FAISS (CPU version)
* Claude and LLaMA generative AI models

## Installation

### Clone the Repository

```bash
git clone TravelGuideChatBotPrototype
cd TravelGuideChatBotPrototype
```

### Setup Environment (Recommended)

Create and activate a conda environment:

```bash
conda create -n travelchat python=3.10 -y
conda activate travelchat
```

Install dependencies:

```bash
pip install -r requirements.txt
```

**Note**: For Apple Silicon users, ensure you install PyTorch optimized for CPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Running the Chatbot

Start the Streamlit application:

```bash
streamlit run src/app.py
```

The chatbot interface will be available at:

```
http://localhost:8501
```

## Project Structure

```
.
├── src
│   ├── app.py           # Streamlit UI entry point
│   ├── instruction_tuning.py   
│   ├── build_faiss_index.py
│   ├── constants.py
│   ├── generate_instruction_data.py
│   └── embeddings.py    
├── faiss_index          # FAISS indexes
├── requirements.txt     # Python dependencies
└── README.md            # This documentation
```

## Future Improvements

* Integration with external travel APIs
* Enhanced chat interactions with larger LLM models
* Support for multilingual queries

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss improvements.