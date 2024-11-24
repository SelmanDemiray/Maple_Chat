Open Source AI Chat
Open Source AI Chat is a powerful, extensible, and easy-to-deploy platform for AI-driven conversations and multimodal interactions. It supports a variety of AI models for text, image, audio, and code generation. The project is designed to be highly customizable and scalable, with Docker Compose for easy deployment and GPU support for high-performance usage.

🚀 Features

🌟 Multimodal AI Model Support
Text Generation: GPT-style models (e.g., GPT-3, Vicuna).
Image Generation: Stable Diffusion, DALL·E.
Audio Generation: Speech-to-text, text-to-speech.
Code Generation: Codex for programming assistance.
Easily switch between models based on tasks.


🌐 Web Search Integration
Enrich AI model responses with real-time web search using the DuckDuckGo API.
Provides more accurate and context-rich answers.
Search can be dynamically enabled or disabled.


⚡ GPU/CPU Support
Full GPU acceleration using NVIDIA CUDA.
Optimized for Apple MPS (Metal Performance Shaders) for macOS.
Automatically falls back to CPU if GPU is unavailable.


🔧 Dynamic Model Management
Switch models dynamically at runtime without rebuilding the container.
Control model configurations via environment variables (MODEL_NAME).


🔐 API Key Authentication
Secure API endpoints with API keys and Secret Keys.
Control access with customizable API rate limits.


⚙️ Easy Deployment with Docker Compose
Simplified local setup and production deployment.
Frontend (Nginx UI) and Backend (FastAPI with AI models) are set up with a single command.
GPU support with nvidia-docker support.
Built-in health checks for monitoring service status.


🖥️ Frontend Integration
Basic Nginx-based frontend serving a simple web UI.
Can be replaced with any custom frontend (React, Vue, etc.).
Static files (index.html, nginx.conf) are easily customizable.


🏥 Health Monitoring
Automatic health checks for both frontend and backend.
Backend checks model loading and inference status.
Frontend ensures Nginx serves the UI correctly.


🛠️ Installation
Prerequisites
Before you begin, make sure you have the following software installed:

Docker: Install Docker
Docker Compose: Install Docker Compose
NVIDIA Drivers (for GPU support): Install NVIDIA Drivers
Setup Instructions


1. Clone the Repository

Copy code
git clone https://github.com/yourusername/open-source-ai-chat.git
cd open-source-ai-chat


2. Configure Environment Variables
Create a .env file in the root of the project with the following content. Adjust the values as needed:

Copy code
# API Configuration
API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here
MODEL_NAME=lmsys/vicuna-7b-v1.5  # Choose your desired model
MAX_TOKENS=1000  # Max tokens for each request
USE_GPU=true  # Set to true if you want GPU support

# Server Configuration
HOST=0.0.0.0
PORT=8007

# Web Search (Optional)
USE_WEB_SEARCH=true  # Enable web search using DuckDuckGo API


3. Build and Run the Application
Run the following command to build and start the application using Docker Compose:

Copy code
docker-compose up --build
This will:

Build the backend service (FastAPI with AI models).
Start the frontend service (Nginx serving the web UI).
Expose the backend API on port 8007 and the frontend UI on port 8080.
🔧 Docker Compose Configuration
The docker-compose.yml file defines two main services: backend (AI service) and frontend (Nginx UI). The configuration includes GPU support and health checks for both services.

Copy code
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
              capabilities: [gpu]  # GPU support
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



🧑‍💻 API Endpoints
POST /chat
Send user input to the AI model and receive a response.

Example Request:

Copy code
curl -X POST "http://localhost:8007/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_input": "What is AI?", "model": "gpt-3.5-turbo"}'

Example Response:

Copy code
{
  "response": "AI stands for Artificial Intelligence, a branch of computer science that aims to create machines capable of performing tasks that typically require human intelligence."
}
GET /health
Check the health of the backend service.

Example Request:

Copy code
curl "http://localhost:8007/health"


Example Response:

Copy code
{
  "status": "healthy"
}


📜 License
MIT License

Copyright (c) 2024 Selman Demiray

Permission is hereby granted, free of charge, to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.





🙌 Contributing
We welcome contributions! If you'd like to improve the project or add new features, please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes.
Commit your changes (git commit -am 'Add new feature').
Push your changes (git push origin feature-branch).
Open a pull request describing your changes.
Please ensure your code is well-tested and documented.


💡 Acknowledgements
FastAPI: Web framework used to build the API.
HuggingFace Transformers: Models for NLP and multimodal applications.
Nginx: Web server for the frontend.
Docker: Containerization platform for deployment.
DuckDuckGo API: Web search capability for real-time context.



⚠️ Disclaimer
This project is provided as-is. We do not provide any warranties or accept liability for damages or legal risks associated with the usage of this software. Make sure to comply with all relevant laws and ethical guidelines when deploying AI models.