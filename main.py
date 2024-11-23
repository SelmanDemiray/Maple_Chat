import logging
import asyncio
import torch
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig, 
    BitsAndBytesConfig
)
from duckduckgo_search import DDGS

# Retain the existing logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enhanced Settings with Open Model Support
class Settings(BaseSettings):
    api_key: str = "default-api-key"
    api_keys: str = "default-api-key"  # Comma-separated list of valid API keys
    secret_key: str = "default-api-key"
    default_model: str = "tiiuae/falcon-7b-instruct"
    max_tokens: int = 500
    host: str = "0.0.0.0"
    port: int = 8007
    debug: bool = False

    # Dictionary of supported models with configurations
    supported_models: Dict[str, Dict[str, str]] = {
        "tiiuae/falcon-7b-instruct": {
            "description": "Falcon 7B Instruct Model",
            "max_tokens": "600",
            "suggested_use": "Interactive conversations and tasks"
        },
        "bigscience/bloom-7b1": {
            "description": "BLOOM 7B Model",
            "max_tokens": "650",
            "suggested_use": "Multilingual text generation"
        },
        "EleutherAI/gpt-neox-20b": {
            "description": "GPT-NeoX 20B Language Model",
            "max_tokens": "800",
            "suggested_use": "Generative text tasks"
        },
        "facebook/opt-6.7b-chat": {
            "description": "OPT 6.7B Chat Model",
            "max_tokens": "550",
            "suggested_use": "Conversational AI"
        },
        "microsoft/DialoGPT-medium": {
            "description": "Microsoft DialoGPT Medium",
            "max_tokens": "500",
            "suggested_use": "Conversational interactions"
        },
        "EleutherAI/pythia-6.9b-dedup": {
            "description": "Pythia 6.9B Model",
            "max_tokens": "550",
            "suggested_use": "Generative tasks"
        },
        "bigcode/starcoder2-3b": {
            "description": "StarCoder2 3B Model",
            "max_tokens": "400",
            "suggested_use": "Coding and technical tasks"
        },
        "stabilityai/stablelm-3b-4e1t": {
            "description": "StableLM 3B Model",
            "max_tokens": "450",
            "suggested_use": "General conversation"
        },
        "mosaicml/mpt-7b": {
            "description": "MPT 7B Model",
            "max_tokens": "600",
            "suggested_use": "Diverse language tasks"
        },
        "allenai/tk-instruct-3b-def": {
            "description": "TK-Instruct 3B Model",
            "max_tokens": "400",
            "suggested_use": "Instruction following"
        }
    }

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()

# Retain existing model classes
class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=2000)

class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = None  # Allow optional model specification
    max_new_tokens: int = Field(default=500, gt=0, le=2000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=5.0)

    @validator('model')
    def validate_model(cls, v, values, **kwargs):
        if v and v not in settings.supported_models:
            raise ValueError(f"Unsupported model. Supported models are: {list(settings.supported_models.keys())}")
        return v or settings.default_model

class ChatResponse(BaseModel):
    response: str
    search_results: Optional[List[dict]] = None

class SearchResult(BaseModel):
    title: str
    snippet: str
    url: str

# Enhanced AI Service with Multi-Model Support
class AIService:
    def __init__(self):
        self.models = {}  # Store multiple model instances
        self.tokenizers = {}  # Store multiple tokenizers
        self.ddgs = None
        self.device = self._get_optimal_device()

    def _get_optimal_device(self):
        """Intelligently select the best available device."""
        if torch.cuda.is_available():
            logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            logger.info("MPS is available. Using Apple Silicon GPU")
            return torch.device("mps")
        else:
            logger.warning("No GPU available. Falling back to CPU.")
            return torch.device("cpu")

    async def initialize(self):
        """Initialize the default model and search service"""
        try:
            await self._load_model(settings.default_model)
            self.ddgs = DDGS()
            logger.info("AI Service initialized successfully.")
        except Exception as e:
            logger.error(f"Critical initialization error: {e}")
            raise RuntimeError(f"Failed to initialize AI service: {e}")

    async def _load_model(self, model_name: str):
        """Load a specific model"""
        logger.info(f"Loading model: {model_name}")
        
        # Skip if already loaded
        if model_name in self.models:
            return

        try:
            # Load Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                use_fast=True,
                trust_remote_code=True
            )
            
            # Determine quantization based on device
            quantization_config = (
                BitsAndBytesConfig(
                    load_in_8bit=True, 
                    llm_int8_threshold=6.0
                ) if self.device.type == 'cuda' else None
            )

            # Load Model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type in ['cuda', 'mps'] else torch.float32,
                device_map='auto',
                quantization_config=quantization_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # Store in dictionaries
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer

        except Exception as e:
            logger.error(f"Model loading failed for {model_name}: {e}")
            raise

    async def get_model(self, model_name: str):
        """Retrieve a specific model, loading if not already initialized"""
        if model_name not in self.models:
            await self._load_model(model_name)
        return self.models[model_name], self.tokenizers[model_name]

    async def search(self, query: str, max_results: int = 3) -> List[SearchResult]:
        """Asynchronous search with thread pooling."""
        def _search():
            try:
                results = []
                for result in self.ddgs.text(query, max_results=max_results):
                    results.append(SearchResult(
                        title=result.get("title", ""),
                        snippet=result.get("body", ""),
                        url=result.get("href", "")
                    ))
                return results
            except Exception as e:
                logger.error(f"Search failed: {e}")
                return []
        
        return await asyncio.to_thread(_search)

    async def generate_response(self, model_name: str, prompt: str, generation_config: GenerationConfig) -> str:
        """Generate response using a specific model"""
        model, tokenizer = await self.get_model(model_name)
        
        try:
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            ).to(self.device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    inputs['input_ids'],
                    generation_config=generation_config
                )
            
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
        
        except Exception as e:
            logger.error(f"Response generation failed for {model_name}: {e}")
            return "I apologize, but I couldn't generate a response."

# FastAPI Application Setup
app = FastAPI(
    title="Multi-Model AI Assistant",
    description="An AI-powered chatbot with web search and dynamic model selection",
    version="1.2.0"
)

ai_service = AIService()

# Add comprehensive CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility Functions
def build_prompt(messages: List[Message], search_results: List[SearchResult]) -> str:
    """Construct a comprehensive prompt with search context."""
    prompt_parts = []
    
    if search_results:
        prompt_parts.append("Contextual Information:")
        for result in search_results[:3]:
            prompt_parts.append(f"- {result.snippet}")
        prompt_parts.append("\nConversation Context:")

    for msg in messages:
        role_prefix = "Human: " if msg.role == "user" else "Assistant: "
        prompt_parts.append(f"{role_prefix}{msg.content}")
    
    return "\n".join(prompt_parts)

# API Key Verification Dependency
async def verify_api_key(x_api_key: str = Header(...)):
    valid_keys = settings.api_keys.split(',')
    if x_api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

# Startup and Shutdown Events
@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Initializing AI Service...")
        await ai_service.initialize()
    except Exception as e:
        logger.critical(f"Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(ai_service, 'ddgs') and ai_service.ddgs:
        ai_service.ddgs.close()
        logger.info("Search client closed.")

# New Endpoint for Model Information
@app.get("/models")
async def get_available_models():
    """Provide information about available models"""
    return {
        "default_model": settings.default_model,
        "models": settings.supported_models
    }

# Main Chat Endpoint
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat_endpoint(request: ChatRequest):
    try:
        # Use the specified model or default
        model_name = request.model or settings.default_model
        
        user_message = next((msg for msg in reversed(request.messages) if msg.role == 'user'), None)
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        search_results = await ai_service.search(user_message.content)
        
        prompt = build_prompt(request.messages, search_results)
        
        generation_config = GenerationConfig(
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            do_sample=True,
            max_new_tokens=request.max_new_tokens
        )
        
        response = await ai_service.generate_response(model_name, prompt, generation_config)
        
        return ChatResponse(
            response=response,
            search_results=[result.dict() for result in search_results]
        )
        
    except Exception as e:
        logger.error(f"Chat processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health Check Endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "default_model": settings.default_model,
        "available_models": list(settings.supported_models.keys()),
        "device": str(ai_service.device)
    }

# Main Execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host=settings.host, 
        port=settings.port, 
        reload=settings.debug
    )


