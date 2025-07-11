# Measles Assistant + Quiz Generator

A RAG-powered medical assistant specialized in measles-related information, built with Together AI's Gemma-3n-E4B-it model.

## Features

- **RAG-powered Q&A**: Answers questions based on WHO measles fact sheet data
- **Quiz Generation**: Creates multiple-choice quizzes from medical context
- **Streaming Responses**: Real-time token streaming for better user experience
- **Caching**: Redis-based caching for improved performance
- **Conversation Memory**: Maintains conversation history for context

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Set your Together AI API key:

```bash
export TOGETHER_API_KEY=your_together_api_key_here
```

### 3. Data Preparation

Ensure the WHO measles fact sheet PDF is in the `data/` directory:
```
data/
└── mm7345a4-H.pdf
```

### 4. Optional: Redis Setup

For caching functionality, install and start Redis:

```bash
# Install Redis (Ubuntu/Debian)
sudo apt-get install redis-server

# Start Redis
sudo systemctl start redis-server

# Or set custom Redis configuration
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=0
```

## Usage

### Run the Application

```bash
python AgentCode.py
```

The application will start a Gradio web interface at `http://localhost:7860`.

### API Endpoints

The application provides two modes:

1. **Answer Mode**: Get factual answers about measles
2. **Quiz Mode**: Generate multiple-choice quizzes

### Example Queries

- "What are the symptoms of measles?"
- "How is measles transmitted?"
- "What is the MMR vaccine?"
- "Generate a quiz about measles prevention"

## Architecture

- **RAG Pipeline**: FAISS vector search + CrossEncoder reranking
- **LLM**: Together AI's Gemma-3n-E4B-it model
- **Embeddings**: LaBSE sentence transformers
- **Caching**: Redis with local fallback
- **UI**: Gradio web interface

## System Prompt

The assistant is configured with a specialized system prompt that:

- Grounds responses in retrieved documents only
- Uses simple, understandable language
- Maintains neutral, educational tone
- Avoids medical diagnoses
- Focuses on public health information

## Error Handling

The application includes comprehensive error handling for:

- Missing API keys
- PDF file not found
- Redis connection issues
- Model generation errors
- JSON parsing errors

## Performance

- **Caching**: Responses cached for 1 hour (Redis) or session (local)
- **Streaming**: Real-time token generation
- **Memory**: Conversation history limited to last 4 exchanges
- **Context**: Top 3 most relevant chunks retrieved per query

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `TOGETHER_API_KEY` is set correctly
2. **PDF Not Found**: Check that `data/mm7345a4-H.pdf` exists
3. **Redis Connection**: Application falls back to local cache if Redis unavailable
4. **Model Errors**: Check Together AI service status and API limits

### Debug Information

The application provides detailed logging:
- `[CACHE]` - Cache hit/miss information
- `[DEBUG]` - Retrieved context details
- `[GEN ERROR]` - Model generation errors
- `[QUIZ ERROR]` - Quiz generation/parsing errors

## License

This project is for educational and research purposes. Please ensure compliance with Together AI's terms of service and data privacy regulations.
