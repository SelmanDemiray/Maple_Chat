import logging
from logging.handlers import RotatingFileHandler
import asyncio
import torch
from typing import List, Optional, Dict, Annotated, Any
from fastapi import FastAPI, HTTPException, Header, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
from contextlib import asynccontextmanager
from functools import lru_cache

import datetime

import os
import sys

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
    CLIPProcessor,
    CLIPModel
)
from duckduckgo_search import DDGS
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import aiohttp
import json
import traceback

# Get log directory from environment variable or use default
LOG_DIR = os.getenv('LOG_DIR', '/app/logs')
LOG_FILE = os.path.join(LOG_DIR, 'app.log')

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Enhanced logging with structured output and rotation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
    ]
)
logger = logging.getLogger(__name__)

# Log startup information
logger.info(f"Starting application with log file: {LOG_FILE}")
logger.info(f"Log directory: {LOG_DIR}")
logger.info(f"Python version: {sys.version}")
logger.info(f"Torch version: {torch.__version__}")

class ModelType(str, Enum):
    """Enhanced model types with string values"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    CODE = "code"

class ModelSize(str, Enum):
    """Enhanced model sizes with string values"""
    TINY = "tiny"  # < 1B parameters
    SMALL = "small"  # 1-3B parameters
    MEDIUM = "medium"  # 3-7B parameters
    LARGE = "large"  # 7-13B parameters
    XLARGE = "xlarge"  # 13B+ parameters

class ModelInfo(BaseModel):
    """Enhanced model information with validation"""
    model_config = ConfigDict(frozen=True)

    description: str
    max_tokens: int  # Changed to int
    suggested_use: str
    type: ModelType
    size: ModelSize
    requires_license: bool = False
    requires_gpu: bool = False
    additional_config: Optional[Dict[str, Any]] = None

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        d = super().dict(*args, **kwargs)
        d["type"] = d["type"].value
        d["size"] = d["size"].value
        return d

@dataclass
class ModelLoadConfig:
    """Configuration for model loading"""
    trust_remote_code: bool = True
    low_cpu_mem_usage: bool = True
    device_map: Optional[str] = None
    torch_dtype: Optional[torch.dtype] = None
    quantization_config: Optional[BitsAndBytesConfig] = None

class Settings(BaseSettings):
    """Enhanced settings with environment variable support"""
    model_config = ConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
        protected_namespaces = ()  # This line updated to fix the warning
    )
    api_key: str = Field(..., env='API_KEY')
    api_keys: str = Field(..., env='API_KEYS')
    secret_key: str = Field(..., env='SECRET_KEY')
    model_name: str = "lmsys/vicuna-7b-v1.5"
    default_model: str = "lmsys/vicuna-7b-v1.5"
    max_tokens: int = 500
    host: str = "0.0.0.0"
    port: int = 8007
    debug: bool = False
    models_config_path: Path = Field(default=Path("models_config.json"))

    # Load supported models from external configuration
    @property
    def supported_models(self) -> Dict[str, ModelInfo]:
        try:
            with open(self.models_config_path) as f:
                models_dict = json.load(f)
            logger.info("Supported Models loaded successfully.")  # Log successful loading
            return {
                k: ModelInfo(**v) for k, v in models_dict.items()
            }
        except Exception as e:
            logger.error(f"Error loading models config: {e}")
            return {}

    def get_models_by_criteria(
        self,
        model_type: Optional[ModelType] = None,
        size: Optional[ModelSize] = None,
        requires_gpu: Optional[bool] = None,
        requires_license: Optional[bool] = None
    ) -> Dict[str, ModelInfo]:
        """Enhanced model filtering with multiple criteria"""
        models = self.supported_models

        if model_type:
            models = {k: v for k, v in models.items() if v.type == model_type}
        if size:
            models = {k: v for k, v in models.items() if v.size == size}
        if requires_gpu is not None:
            models = {k: v for k, v in models.items() if v.requires_gpu == requires_gpu}
        if requires_license is not None:
            models = {k: v for k, v in models.items() if v.requires_license == requires_license}

        return models

class Message(BaseModel):
    """Enhanced message model with role validation"""
    model_config = ConfigDict(str_strip_whitespace=True)

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=2000)

class ChatRequest(BaseModel):
    """Enhanced chat request with comprehensive validation"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        populate_by_name=True
    )

    messages: List[Message]
    model: Optional[str] = None
    max_new_tokens: int = Field(default=500, gt=0, le=2000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=5.0)

    @field_validator('model')
    @classmethod
    def validate_model(cls, v: Optional[str], info: Dict[str, Any]) -> str:
        settings = get_settings()
        if v and v not in settings.supported_models:
            raise ValueError(f"Unsupported model. Available models: {list(settings.supported_models.keys())}")
        return v or settings.default_model

class ChatResponse(BaseModel):
    """Enhanced response model with metadata"""
    model_config = ConfigDict(str_strip_whitespace=True)

    response: str
    search_results: Optional[List[dict]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchResult(BaseModel):
    """Enhanced search result model"""
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str
    snippet: str
    url: str
    relevance_score: Optional[float] = None

class ImageRequest(BaseModel):
    """Request model for image generation"""
    model_config = ConfigDict(str_strip_whitespace=True)

    prompt: str = Field(..., min_length=1, max_length=2000)
    model: Optional[str] = None
    num_images: int = Field(default=1, ge=1, le=4)

    @field_validator('model')
    @classmethod
    def validate_model(cls, v: Optional[str], info: Dict[str, Any]) -> str:
        settings = get_settings()
        if v and v not in settings.supported_models:
            raise ValueError(f"Unsupported model. Available models: {list(settings.supported_models.keys())}")
        return v or settings.default_model

class ImageResponse(BaseModel):
    """Response model for image generation"""
    model_config = ConfigDict(str_strip_whitespace=True)

    images: List[str] = Field(...)  # List of base64 encoded images
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AIService:
    """Enhanced AI service with improved error handling and resource management"""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.image_processors: Dict[str, Any] = {}
        self.ddgs: Optional[DDGS] = None
        self.device = self._get_optimal_device()
        self.current_model_name: Optional[str] = None
        self._http_session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    @staticmethod
    def _get_optimal_device() -> torch.device:
        """Enhanced device selection with MPS support"""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Using CUDA GPU: {device_name}")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            logger.info("Using Apple Silicon GPU (MPS)")
            return torch.device("mps")
        else:
            logger.warning("No GPU available, using CPU")
            return torch.device("cpu")

    def _get_model_config(self, model_name: str) -> ModelLoadConfig:
        """Enhanced model configuration with dataclass and improved dtype handling"""
        base_config = ModelLoadConfig(
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        if self.device.type == 'cuda':
            base_config.quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_enable_fp32_cpu_offload=True
            )
            base_config.device_map = "auto"

            # Use auto detection of dtype instead of forcing float16
            base_config.torch_dtype = None
        else:
            base_config.torch_dtype = torch.float32

        return base_config

    async def initialize(self):
        """Enhanced initialization with connection pooling"""
        try:
            self._http_session = aiohttp.ClientSession()
            logger.info(f"Initializing with default model: {get_settings().default_model}")
            await self._load_model(get_settings().default_model)
        except Exception as e:
            logger.error(f"Initialization failed: {traceback.format_exc()}")
            await self.cleanup()
            raise

    async def _load_model(self, model_name: str):
        """Enhanced model loading with better resource management"""
        if model_name in self.models:
            if model_name != self.current_model_name:
                await self._unload_current_model()
            return

        try:
            config = self._get_model_config(model_name)
            model_info = get_settings().supported_models[model_name]

            if model_info.type == ModelType.TEXT:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=config.trust_remote_code,
                    low_cpu_mem_usage=config.low_cpu_mem_usage,
                    device_map=config.device_map,
                    torch_dtype=config.torch_dtype,
                    quantization_config=config.quantization_config
                )
                if config.device_map is None:
                    model = model.to(self.device)

                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
            elif model_info.type == ModelType.IMAGE:
                processor = CLIPProcessor.from_pretrained(model_name)
                model = CLIPModel.from_pretrained(model_name)
                model = model.to(self.device)

                self.models[model_name] = model
                self.image_processors[model_name] = processor
            else:
                raise ValueError(f"Unsupported model type: {model_info.type}")

            self.current_model_name = model_name

            logger.info(f"Model {model_name} loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Model loading error: {traceback.format_exc()}")
            raise

    async def generate_response(
        self,
        model_name: str,
        prompt: str,
        generation_config: GenerationConfig
    ) -> str:
        """Enhanced response generation with better error handling"""
        try:
            if model_name != self.current_model_name:
                await self._load_model(model_name)

            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()

        except Exception as e:
            logger.error(f"Generation error: {traceback.format_exc()}")
            raise

    async def generate_image(
        self,
        model_name: str,
        prompt: str,
        num_images: int
    ) -> List[str]:
        """Enhanced image generation with better error handling"""
        try:
            if model_name != self.current_model_name:
                await self._load_model(model_name)

            processor = self.image_processors[model_name]
            model = self.models[model_name]

            inputs = processor(text=prompt, return_tensors="pt").to(self.device)

            with torch.inference_mode():
                images = model.generate(**inputs, num_images_per_prompt=num_images)

            # Convert images to base64 for easy transfer
            base64_images = []
            for image in images:
                import io
                from PIL import Image
                import base64

                buffered = io.BytesIO()
                Image.fromarray(image.cpu().numpy()).save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                base64_images.append(img_str)

            return base64_images

        except Exception as e:
            logger.error(f"Image generation error: {traceback.format_exc()}")
            raise

    async def search(
        self,
        query: str,
        max_results: int = 3
    ) -> List[SearchResult]:
        """Enhanced search with async implementation and improved error handling"""
        try:
            if self.ddgs is None:
                self.ddgs = DDGS()

            results = []
            ddg_results = list(self.ddgs.text(query, max_results=max_results))

            for r in ddg_results:
                # Use .get() to safely extract keys, with fallback values
                results.append(SearchResult(
                    title=r.get('title', 'No Title'),
                    snippet=r.get('body', 'No snippet available'),
                    url=r.get('link') or r.get('href') or r.get('url', ''),
                    relevance_score=0.0  # future relevance scoring
                ))

            return results

        except Exception as e:
            logger.error(f"Search error: {traceback.format_exc()}")
            return []

    async def cleanup(self):
        """Enhanced cleanup with comprehensive resource management"""
        try:
            await self._unload_current_model()
            if self.ddgs:
                self.ddgs.close()
            if self._http_session:
                await self._http_session.close()
        except Exception as e:
            logger.error(f"Cleanup error: {traceback.format_exc()}")

    async def _unload_current_model(self):
        """Enhanced model unloading with explicit device removal"""
        if self.current_model_name:
            try:
                model_info = get_settings().supported_models[self.current_model_name]
                if model_info.type == ModelType.TEXT:
                    del self.models[self.current_model_name]
                    del self.tokenizers[self.current_model_name]
                elif model_info.type == ModelType.IMAGE:
                    del self.models[self.current_model_name]
                    del self.image_processors[self.current_model_name]
                else:
                    raise ValueError(f"Unsupported model type: {model_info.type}")

                # Explicitly move model to CPU before deleting to free GPU memory
                # torch.cuda.empty_cache()
                self.current_model_name = None
                logger.info(f"Unloaded model: {self.current_model_name}")
            except Exception as e:
                logger.error(f"Error unloading model: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global ai_service
    ai_service = AIService()
    try:
        await ai_service.initialize()
        yield
    finally:
        await ai_service.cleanup()

# Enhanced FastAPI application with OpenAPI documentation
app = FastAPI(
    title="Multi-Model AI Assistant",
    description="Advanced AI-powered chatbot with web search and dynamic model selection",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@lru_cache()
def get_settings():
    """Cached settings provider"""
    return Settings()

async def verify_api_key(
    api_key: Annotated[str, Header(alias="X-API-Key")]
) -> str:
    """Enhanced API key verification"""
    settings = get_settings()
    valid_keys = settings.api_keys.split(',')
    if api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    return api_key

def build_prompt(
    messages: List[Message],
    search_results: List[SearchResult]
) -> str:
    """Enhanced prompt builder with better formatting"""
    parts = []

    if search_results:
        parts.append("### Contextual Information:")
        parts.extend(f"- {result.snippet}" for result in search_results[:3])
        parts.append("\n### Conversation:")

    parts.extend(
        f"{'Human' if msg.role == 'user' else 'Assistant'}: {msg.content}"
        for msg in messages
    )

    return "\n".join(parts)

@app.get("/models", response_model=Dict[str, Any])
async def get_available_models(
    model_type: Optional[ModelType] = None,
    size: Optional[ModelSize] = None,
    requires_gpu: Optional[bool] = None
):
    """Enhanced model information endpoint with filtering"""
    settings = get_settings()

    models = settings.get_models_by_criteria(
        model_type=model_type,
        size=size,
        requires_gpu=requires_gpu
    )

    return {
        "models": models,
        "default_model": settings.default_model
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Enhanced chat endpoint with background tasks and improved error handling"""
    try:
        model_name = request.model or get_settings().default_model
        model_info = get_settings().supported_models.get(model_name)

        if model_info.requires_gpu and ai_service.device.type == 'cpu':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model '{model_name}' requires a GPU, but no GPU is available."
            )

        search_results = await ai_service.search(request.messages[-1].content)
        prompt = build_prompt(request.messages, search_results)

        generation_config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            do_sample=True  # Ensure sampling is enabled
        )

        response = await ai_service.generate_response(
            model_name,
            prompt,
            generation_config
        )

        # Add a background task to log the chat interaction
        background_tasks.add_task(log_chat_interaction, request, response, search_results)

        return ChatResponse(
            response=response,
            search_results=[result.dict() for result in search_results],
            metadata={
                "model_name": model_name,
                "timestamp": datetime.datetime.now().isoformat()
            }
        )

    except Exception as e:
        logger.error(f"Chat processing error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing chat request."
        ) from e

@app.post("/image", response_model=ImageResponse)
async def image(
    request: ImageRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Enhanced image generation endpoint with background tasks and improved error handling"""
    try:
        model_name = request.model or get_settings().default_model
        model_info = get_settings().supported_models.get(model_name)

        if model_info.requires_gpu and ai_service.device.type == 'cpu':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model '{model_name}' requires a GPU, but no GPU is available."
            )

        images = await ai_service.generate_image(
            model_name,
            request.prompt,
            request.num_images
        )

        # Add a background task to log the image generation request
        background_tasks.add_task(log_image_generation, request, images)

        return ImageResponse(
            images=images,
            metadata={
                "model_name": model_name,
                "timestamp": datetime.datetime.now().isoformat()
            }
        )

    except Exception as e:
        logger.error(f"Image generation error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating image."
        ) from e

def log_chat_interaction(
    request: ChatRequest,
    response: str,
    search_results: List[SearchResult]
):
    """Enhanced logging with structured data and search results"""
    try:
        log_data = {
            "model": request.model,
            "prompt": build_prompt(request.messages, search_results),
            "response": response,
            "search_results": [result.dict() for result in search_results],
            "timestamp": datetime.datetime.now().isoformat()
        }
        logger.info(f"Chat interaction: {json.dumps(log_data)}")
    except Exception as e:
        logger.error(f"Error logging chat interaction: {e}")

def log_image_generation(
    request: ImageRequest,
    images: List[str]
):
    """Enhanced logging for image generation requests"""
    try:
        log_data = {
            "model": request.model,
            "prompt": request.prompt,
            "num_images": request.num_images,
            "timestamp": datetime.datetime.now().isoformat()
        }
        logger.info(f"Image generation request: {json.dumps(log_data)}")
    except Exception as e:
        logger.error(f"Error logging image generation: {e}")


@app.get("/hardware")
async def get_hardware_info():
    """
    Endpoint to return detailed hardware information and log it.
    """
    hardware_info = {}

    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            hardware_info["gpu"] = device_name.lower()  # e.g., "nvidia geforce rtx 3080"
            hardware_info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
        elif torch.backends.mps.is_available():
            hardware_info["gpu"] = "apple silicon gpu"
        else:
            hardware_info["gpu"] = "cpu"

        # Add CPU information
        hardware_info["cpu_cores"] = os.cpu_count()

        # Add RAM information
        import psutil
        mem = psutil.virtual_memory()
        hardware_info["ram_total"] = mem.total
        hardware_info["ram_available"] = mem.available

        # Log hardware information
        logger.info(f"Hardware information: {hardware_info}")

    except Exception as e:
        logger.error(f"Error retrieving hardware information: {e}")

    return hardware_info

@app.get("/ping") 
async def ping():
    """
    Simple ping endpoint to check server availability and latency.
    """
    return {"message": "pong"}

# Health Check Endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "default_model": get_settings().default_model,  # Use get_settings()
        "available_models": list(get_settings().supported_models.keys()),  # Use get_settings()
        "device": str(ai_service.device)
    }


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)