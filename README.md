Open Source AI Chat
Open Source AI Chat is an extensible, high-performance platform designed for AI-driven conversations and multimodal interactions. It supports a variety of AI models for text, image, audio, and code generation, offering flexibility and scalability. The project is optimized for easy deployment using Docker Compose and supports GPU acceleration for high-performance usage.

🚀 Features
🌟 Multimodal AI Model Support
Text Generation: GPT-style models (e.g., GPT-3, Vicuna).
Image Generation: Powered by models like Stable Diffusion and DALL·E.
Audio Generation: Includes speech-to-text and text-to-speech functionality.
Code Generation: Leverage Codex for programming assistance.
Seamlessly switch between models based on specific tasks.
🌐 Web Search Integration
Real-Time Search: Enhance model responses with live data from the DuckDuckGo API.
Dynamic Enablement: Toggle search functionality based on your needs.
⚡ GPU/CPU Support
GPU Acceleration: Full support for NVIDIA CUDA.
Apple MPS Support: Optimized for Apple devices (Metal Performance Shaders).
Fallback to CPU: Automatically switches to CPU if GPU is unavailable.
🔧 Dynamic Model Management
Runtime Model Switching: Change models without rebuilding the container.
Environment Variables: Easily configure model settings via environment variables (MODEL_NAME, etc.).
🔐 API Key Authentication
Secure your endpoints with API and secret keys.
Customizable Rate Limits: Control access with flexible API rate limits.
⚙️ Easy Deployment with Docker Compose
One-Command Setup: Deploy both frontend (Nginx UI) and backend (FastAPI with AI models) services in a single step.
GPU Support: Docker integration for NVIDIA GPU acceleration.
Health Monitoring: Built-in checks for both frontend and backend services.
🖥️ Frontend Integration
Basic Web UI: Frontend powered by Nginx, serving a simple interface.
Customizable: Replace the UI with your preferred frontend framework (React, Vue, etc.).
🏥 Health Monitoring
Automated Health Checks: Monitors both backend (AI models) and frontend (Nginx) services.
Status Reporting: Provides clear insights into service health.
🛠️ Installation
Prerequisites
Make sure you have the following installed:

Docker: Install Docker
Docker Compose: Install Docker Compose
NVIDIA Drivers (for GPU support): Install NVIDIA Drivers
Setup Instructions
Clone the Repository
bash
Copy code
git clone https://github.com/yourusername/open-source-ai-chat.git
cd open-source-ai-chat
Configure Environment Variables Create a .env file in the root of the project with the following configuration. Adjust values as necessary:
bash
Copy code
# API Configuration
API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here
MODEL_NAME=lmsys/vicuna-7b-v1.5  # Choose your model
MAX_TOKENS=1000  # Max tokens per request
USE_GPU=true  # Set to 'true' for GPU support

# Server Configuration
HOST=0.0.0.0
PORT=8007

# Web Search (Optional)
USE_WEB_SEARCH=true  # Enable DuckDuckGo search
Build and Run the Application Run the following command to build and launch the application using Docker Compose:
bash
Copy code
docker-compose up --build
This will:

Build and start the backend service (FastAPI with AI models).
Start the frontend service (Nginx serving the web UI).
Expose the backend API on port 8007 and the frontend UI on port 8080.
🔧 Docker Compose Configuration
The docker-compose.yml file defines two services: backend (AI service) and frontend (Nginx UI). GPU support is enabled, and health checks are configured for both services.

yaml
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
              capabilities: [gpu]  # Enable GPU support
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
Send user input to the AI model and get a response.

Example Request:

bash
Copy code
curl -X POST "http://localhost:8007/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_input": "What is AI?", "model": "gpt-3.5-turbo"}'
Example Response:

json
Copy code
{
  "response": "AI stands for Artificial Intelligence, a branch of computer science that aims to create machines capable of performing tasks that typically require human intelligence."
}
GET /health
Check the health of the backend service.

Example Request:

bash
Copy code
curl "http://localhost:8007/health"
Example Response:

json
Copy code
{
  "status": "healthy"
}
📜 License
This project is licensed under the MIT License.

🙌 Contributing
We welcome contributions! To improve the project or add new features:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes.
Commit your changes (git commit -am 'Add new feature').
Push your changes (git push origin feature-branch).
Open a pull request and describe your changes.
Ensure your code is well-tested and documented.

💡 Acknowledgements
FastAPI: Web framework for building the API.
HuggingFace Transformers: Models for NLP and multimodal applications.
Nginx: Web server for the frontend UI.
Docker: Containerization platform for deployment.
DuckDuckGo API: Web search for real-time context.
⚠️ Disclaimer
This project is provided "as-is" with no warranty. Ensure compliance with all legal and ethical guidelines when deploying AI models. We are not liable for any damages or legal risks associated with its usage.