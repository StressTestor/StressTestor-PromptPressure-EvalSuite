"""
Adapter registry for PromptPressure Eval Suite v1.5.2

This module provides a registry of available model adapters and their corresponding
generate_response functions. Each adapter handles communication with a specific
LLM provider (Groq, LM Studio, OpenAI, etc.) and exposes a consistent interface.

Available adapters:
- groq: For Groq's cloud-based LLM services
- lmstudio: For local LM Studio inference server
- openai: For OpenAI's API services
- mock: For testing and development without external API calls

Example usage:
    from adapters import ADAPTER_REGISTRY
    
    # Get the appropriate adapter
    adapter_fn = ADAPTER_REGISTRY["openai"]
    
    # Generate a response
    response = adapter_fn("Hello, world!", "gpt-4", {"temperature": 0.7})
"""

from typing import Dict, Callable, Any
from .groq_adapter     import generate_response as groq_generate_response
from .lmstudio_adapter import generate_response as lmstudio_generate_response
from .openai_adapter   import generate_response as openai_generate_response
from .mock_adapter     import generate_response as mock_generate_response

# Main registry of available adapters
# Each adapter must implement the generate_response function with the signature:
#   generate_response(prompt: str, model_name: str, config: Dict[str, Any]) -> Union[str, List[str], Iterator[str]]
ADAPTER_REGISTRY: Dict[str, Callable[..., Any]] = {
    "groq": groq_generate_response,
    "lmstudio": lmstudio_generate_response,
    "openai": openai_generate_response,
    "mock": mock_generate_response
}

# Backwards compatibility alias for v1.4
ADAPTER_REGISTRY_V1_4 = ADAPTER_REGISTRY

# Current version of the adapters package
__version__ = "1.5.2"
