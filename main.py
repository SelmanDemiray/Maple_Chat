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
from enum import Enum

from typing import List



# Retain the existing logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    CODE = "code"

class ModelSize(Enum):
    TINY = "tiny"      # < 1B parameters
    SMALL = "small"    # 1-3B parameters
    MEDIUM = "medium"  # 3-7B parameters
    LARGE = "large"    # 7-13B parameters
    XLARGE = "xlarge"  # 13B+ parameters

class ModelInfo(BaseModel):
    description: str
    max_tokens: str
    suggested_use: str
    type: ModelType
    size: ModelSize
    requires_license: bool = False
    requires_gpu: bool = False
    additional_config: Optional[Dict] = None

class Settings(BaseSettings):
    api_key: str = "default-api-key"
    api_keys: str = "default-api-key"  # Comma-separated list of valid API keys
    secret_key: str = "default-api-key"
    model_name: str = "lmsys/vicuna-7b-v1.5"
    default_model: str = "lmsys/vicuna-7b-v1.5"
    max_tokens: int = 500
    host: str = "0.0.0.0"
    port: int = 8007
    debug: bool = False

    # Comprehensive model configurations
    supported_models: Dict[str, ModelInfo] = {
        # Large Language Models - Small Size (1-3B)
        "lmsys/vicuna-7b-v1.5": {
            "description": "Vicuna-7B - Open-source chat model",
            "max_tokens": "4096",
            "suggested_use": "General-purpose chat application",
            "type": ModelType.TEXT,
            "size": ModelSize.LARGE,
            "requires_gpu": True
        },
        "lmsys/vicuna-13b-v1.5": {
            "description": "Vicuna-13B - Larger, more capable version of Vicuna",
            "max_tokens": "8192",
            "suggested_use": "High-performance chat applications",
            "type": ModelType.TEXT,
            "size": ModelSize.XLARGE,
            "requires_gpu": True
        },
        "huggingface/llama-2-7b-chat": {
            "description": "Llama-2 7B - Open-source chat model by Meta",
            "max_tokens": "4096",
            "suggested_use": "General-purpose conversational agent",
            "type": ModelType.TEXT,
            "size": ModelSize.LARGE,
            "requires_gpu": True
        },
        "huggingface/llama-2-13b-chat": {
            "description": "Llama-2 13B - More powerful version of Llama",
            "max_tokens": "8192",
            "suggested_use": "High-performance conversation and NLP tasks",
            "type": ModelType.TEXT,
            "size": ModelSize.XLARGE,
            "requires_gpu": True
        },
        "microsoft/phi-2": {
            "description": "Microsoft Phi-2 - Compact but powerful 2.7B model",
            "max_tokens": "2048",
            "suggested_use": "General text and code generation",
            "type": ModelType.TEXT,
            "size": ModelSize.SMALL,
            "requires_gpu": False
        },
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
            "description": "TinyLlama - Efficient 1.1B chat model",
            "max_tokens": "2048",
            "suggested_use": "Lightweight chat applications",
            "type": ModelType.TEXT,
            "size": ModelSize.TINY,
            "requires_gpu": False
        },
        "stabilityai/stablelm-zephyr-3b": {
            "description": "StableLM Zephyr 3B - Instruction-tuned model",
            "max_tokens": "4096",
            "suggested_use": "General conversation",
            "type": ModelType.TEXT,
            "size": ModelSize.SMALL,
            "requires_gpu": True
        },
        # Code Generation Models
        "bigcode/starcoderbase-3b": {
            "description": "StarCoderBase 3B - Lightweight code model",
            "max_tokens": "4096",
            "suggested_use": "Code generation for smaller deployments",
            "type": ModelType.CODE,
            "size": ModelSize.SMALL,
            "requires_gpu": False
        },

        # Image Generation Models
        "stabilityai/sdxl-turbo": {
            "description": "SDXL Turbo - Fast image generation",
            "max_tokens": "N/A",
            "suggested_use": "Real-time image generation",
            "type": ModelType.IMAGE,
            "size": ModelSize.LARGE,
            "requires_gpu": True,
            "additional_config": {
                "image_size": 1024,
                "steps": 1
            }
        },
        "stabilityai/stable-diffusion-xl-base-1.0": {
            "description": "Stable Diffusion XL 1.0 - High quality image generation",
            "max_tokens": "N/A",
            "suggested_use": "High-quality image creation",
            "type": ModelType.IMAGE,
            "size": ModelSize.LARGE,
            "requires_gpu": True,
            "additional_config": {
                "image_size": 1024,
                "steps": 30
            }
        },
        "runwayml/stable-diffusion-v1-5": {
            "description": "Stable Diffusion 1.5 - Balanced image generation",
            "max_tokens": "N/A",
            "suggested_use": "General purpose image generation",
            "type": ModelType.IMAGE,
            "size": ModelSize.MEDIUM,
            "requires_gpu": True,
            "additional_config": {
                "image_size": 512,
                "steps": 20
            }
        },

        # Video Generation Models
        "stabilityai/stable-video-diffusion-img2vid": {
            "description": "Stable Video Diffusion - Image to video generation",
            "max_tokens": "N/A",
            "suggested_use": "Converting images to short videos",
            "type": ModelType.VIDEO,
            "size": ModelSize.LARGE,
            "requires_gpu": True,
            "additional_config": {
                "max_frames": 25,
                "fps": 8
            }
        },
        "damo-vilab/text-to-video-ms-1.7b": {
            "description": "ModelScope Text2Video - Text to video generation",
            "max_tokens": "N/A",
            "suggested_use": "Generating videos from text descriptions",
            "type": ModelType.VIDEO,
            "size": ModelSize.MEDIUM,
            "requires_gpu": True,
            "additional_config": {
                "max_frames": 16,
                "fps": 8
            }
        },

        # Specialized Language Models
        "openchat/openchat-3.5-0106": {
            "description": "OpenChat 3.5 - Open source chat model",
            "max_tokens": "8192",
            "suggested_use": "Open source chat alternative",
            "type": ModelType.TEXT,
            "size": ModelSize.LARGE,
            "requires_gpu": True
        },
        "BAAI/bge-large-en-v1.5": {
            "description": "BGE Large - Text embeddings model",
            "max_tokens": "512",
            "suggested_use": "Text embeddings and similarity",
            "type": ModelType.TEXT,
            "size": ModelSize.LARGE,
            "requires_gpu": False
        }
    }

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

    def get_models_by_type(self, model_type: ModelType) -> Dict[str, ModelInfo]:
        """Get all models of a specific type"""
        return {k: v for k, v in self.supported_models.items() if v.type == model_type}

    def get_models_by_size(self, size: ModelSize) -> Dict[str, ModelInfo]:
        """Get all models of a specific size"""
        return {k: v for k, v in self.supported_models.items() if v.size == size}

    def get_gpu_required_models(self) -> Dict[str, ModelInfo]:
        """Get all models that require GPU"""
        return {k: v for k, v in self.supported_models.items() if v.requires_gpu}

    def get_licensed_models(self) -> Dict[str, ModelInfo]:
        """Get all models that require licenses"""
        return {k: v for k, v in self.supported_models.items() if v.requires_license}
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
        self.models = {}
        self.tokenizers = {}
        self.ddgs = None
        self.device = self._get_optimal_device()
        self.current_model_name = None
        
    def _get_optimal_device(self):
        if torch.cuda.is_available():
            logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            logger.info("MPS is available. Using Apple Silicon GPU")
            return torch.device("mps")
        else:
            logger.warning("No GPU available. Falling back to CPU.")
            return torch.device("cpu")

    def _get_model_config(self, model_name: str) -> dict:
        """Get the configuration for loading a specific model"""
        base_config = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        # Add device configuration based on available hardware
        if self.device.type == 'cuda':
            # Create quantization config for CUDA devices
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            base_config.update({
                "device_map": "auto",
                "quantization_config": quantization_config
            })
        
        # Get model info from settings
        model_info = settings.supported_models.get(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found in supported models")

        # Add any model-specific configurations
        if model_info.additional_config:
            base_config.update(model_info.additional_config)

        return base_config

    async def initialize(self):
        """Initialize the AI service and load the default model"""
        try:
            logger.info(f"Initializing service with default model: {settings.default_model}")
            await self._load_model(settings.default_model)
        except Exception as e:
            logger.error(f"Failed to initialize AI service: {e}")
            raise

    async def _load_model(self, model_name: str):
        """Load a model and move it to the correct device"""
        if model_name in self.models:
            if model_name != self.current_model_name:
                await self._unload_current_model()
            return

        try:
            # Get model configuration
            model_config = self._get_model_config(model_name)
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_config,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            )

            # Move model to device if not using device_map
            if 'device_map' not in model_config:
                model = model.to(self.device)

            # Store model and tokenizer
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            self.current_model_name = model_name
            
            logger.info(f"Model {model_name} loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise

    async def generate_response(self, model_name: str, prompt: str, generation_config: GenerationConfig):
        """Generate a response using the specified model"""
        try:
            # Ensure the correct model is loaded
            if model_name != self.current_model_name:
                await self._load_model(model_name)

            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]

            # Prepare inputs on the correct device
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()  # Remove prompt from response
            
            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    async def _unload_current_model(self):
        """Safely unload the current model"""
        if self.current_model_name:
            try:
                if self.current_model_name in self.models:
                    del self.models[self.current_model_name]
                if self.current_model_name in self.tokenizers:
                    del self.tokenizers[self.current_model_name]
                torch.cuda.empty_cache()
                self.current_model_name = None
                logger.info("Successfully unloaded current model")
            except Exception as e:
                logger.error(f"Error unloading model: {e}")
                raise

    async def search(self, query: str, max_results: int = 3) -> List[SearchResult]:
        """Perform web search and return results"""
        try:
            if self.ddgs is None:
                self.ddgs = DDGS()

            search_results = []
            async for r in self.ddgs.text(query, max_results=max_results):
                search_results.append(SearchResult(
                    title=r['title'],
                    snippet=r['body'],
                    url=r['link']
                ))
            return search_results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []  # Return empty list on error

    async def cleanup(self):
        """Cleanup resources"""
        await self._unload_current_model()
        if self.ddgs:
            self.ddgs.close()

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


