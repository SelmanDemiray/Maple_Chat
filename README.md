![Logo](./logo.png)
# Open Source AI Chat 🤖

A powerful, extensible, and easy-to-deploy platform for AI-driven conversations and multimodal interactions. This platform supports various AI models for text, image, audio, and code generation, designed with customization and scalability in mind.

## ✨ Key Features

### 🌟 Multimodal AI Model Support
- **Text Generation**: GPT-style models (GPT-3, Vicuna)
- **Image Generation**: Stable Diffusion, DALL·E
- **Audio Processing**: Speech-to-text, text-to-speech capabilities
- **Code Generation**: Codex for programming assistance
- Flexible model switching based on specific tasks

### 🌐 Web Search Integration
- Real-time web search enrichment using DuckDuckGo API
- Enhanced context-rich responses
- Configurable search functionality

### ⚡ Performance
- Full GPU acceleration with NVIDIA CUDA support
- Optimized for Apple MPS (Metal Performance Shaders) on macOS
- Automatic CPU fallback when GPU is unavailable

### 🔧 Advanced Features
- **Dynamic Model Management**: Runtime model switching without container rebuilds
- **API Security**: Robust authentication with API keys and customizable rate limits
- **Health Monitoring**: Automatic checks for frontend and backend services

![Screenshot](/screenshot.png)


## 🚀 Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx) (for GPU support)

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/open-source-ai-chat.git
   cd open-source-ai-chat
   ```

2. **Configure Environment**
   Create a `.env` file in the project root:
   ```env
   # API Configuration
   API_KEY=your_api_key_here
   SECRET_KEY=your_secret_key_here
   MODEL_NAME=lmsys/vicuna-7b-v1.5
   MAX_TOKENS=1000
   USE_GPU=true

   # Server Configuration
   HOST=0.0.0.0
   PORT=8007

   # Web Search Configuration
   USE_WEB_SEARCH=true
   ```

3. **Launch the Application**
   ```bash
   docker-compose up --build
   ```

## 🔌 API Reference

### Chat Endpoint
```bash
POST /chat

# Example Request
curl -X POST "http://localhost:8007/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_input": "What is AI?", "model": "gpt-3.5-turbo"}'

# Example Response
{
  "response": "AI stands for Artificial Intelligence, a branch of computer science..."
}
```

### Health Check
```bash
GET /health

# Example Response
{
  "status": "healthy"
}
```

## 🐳 Docker Configuration

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - API_KEY=${API_KEY}
      - SECRET_KEY=${SECRET_KEY}
      - MODEL_NAME=${MODEL_NAME}
      - MAX_TOKENS=${MAX_TOKENS}
      - HOST=${HOST}
      - PORT=${PORT}
    env_file:
      - .env
    ports:
      - "8007:8007"
    volumes:
      - huggingface_cache:/home/myuser/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8007/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  frontend:
    image: nginx:alpine
    volumes:
      - ./index.html:/usr/share/nginx/html/index.html
      - ./nginx.conf:/etc/nginx/nginx.conf
    ports:
      - "8080:80"
    depends_on:
      - backend
    healthcheck:
      test: ["CMD", "wget", "-q", "-O", "-", "http://localhost:80/"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  huggingface_cache:
    driver: local
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Technologies Used

- [FastAPI](https://fastapi.tiangolo.com/): Web framework
- [HuggingFace Transformers](https://huggingface.co/): AI models
- [Nginx](https://nginx.org/): Web server
- [Docker](https://www.docker.com/): Containerization
- [DuckDuckGo API](https://duckduckgo.com/): Web search integration

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

## ⚠️ Disclaimer

This project is provided as-is without warranties. Users are responsible for ensuring compliance with relevant laws and ethical guidelines when deploying AI models.