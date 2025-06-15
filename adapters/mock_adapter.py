
"""
Mock adapter for PromptPressure Eval Suite v1.5.2

Provides deterministic placeholder responses for testing and development.
Useful for CI, local testing, and development without external API calls.
"""

from typing import Optional, Dict, Any, Union, List, Iterator
import time
import random
import logging

# Set up logger
logger = logging.getLogger(__name__)

class MockError(Exception):
    """Base exception for mock adapter errors."""
    pass


def _generate_mock_response(
    prompt: str,
    model_name: str,
    config: Dict[str, Any]
) -> str:
    """Generate a mock response based on the input and configuration."""
    response_parts = [
        f"[SIMULATED RESPONSE – PromptPressure Mock Adapter v1.5.2]\n\n",
        f"Model: {model_name}\n",
        f"Temperature: {config.get('temperature')}\n",
        f"Max tokens: {config.get('max_tokens', 'unlimited')}\n\n",
        "This is a mocked reply for testing the evaluation pipeline.\n\n",
        "Prompt received (first 100 chars):\n---\n",
        f"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\n---\n"
    ]
    
    # Add some variability based on config
    if config.get("echo", False):
        response_parts.append(f"\nFull prompt (echoed back):\n---\n{prompt}\n---\n")
    
    response_parts.append("\n(No external API call was made.)")
    return "".join(response_parts)


def _stream_mock_response(
    prompt: str,
    model_name: str,
    config: Dict[str, Any]
) -> Iterator[str]:
    """Generate a mock streaming response."""
    # Set random seed if provided for reproducibility
    if config.get("mock_seed") is not None:
        random.seed(config["mock_seed"])
    
    full_response = _generate_mock_response(prompt, model_name, config)
    words = full_response.split()
    
    # Validate and clamp delay
    delay = config.get("mock_stream_delay", 0.05)
    try:
        delay = float(delay)
        if delay <= 0:
            raise ValueError
    except Exception:
        logger.warning(f"Invalid mock_stream_delay: {delay}. Using default 0.05s.")
        delay = 0.05
    
    for i, word in enumerate(words):
        chunk = word + ("" if i == len(words) - 1 else " ")
        # 30% chance to split the word
        if random.random() < 0.3 and len(word) > 1:
            mid = len(word) // 2
            yield word[:mid]
            time.sleep(delay)
            yield word[mid:] + ("" if i == len(words) - 1 else " ")
        else:
            yield chunk
        time.sleep(delay)


def generate_response(
    prompt: str,
    model_name: str = "mock-model",
    config: Optional[Dict[str, Any]] = None
) -> Union[str, List[str], Iterator[str]]:
    """
    Generate a mock response for testing purposes.
    """
    if not prompt or not isinstance(prompt, str):
        raise MockError("Prompt must be a non-empty string")
    
    # Merge with default config and validate
    merged_config = {
        "temperature": 0.7,
        "max_tokens": None,
        "n": 1,
        "stream": False,
        "echo": False,
        "mock_stream_delay": 0.05,
        "mock_seed": None,
        **(config or {})
    }
    # Clamp and validate temperature
    try:
        temp = float(merged_config.get("temperature", 0.7))
        if not 0.0 <= temp <= 2.0:
            logger.warning(f"Temperature {temp} clamped to [0.0, 2.0]")
            temp = max(0.0, min(2.0, temp))
        merged_config["temperature"] = temp
    except Exception:
        logger.warning(f"Invalid temperature: {merged_config.get('temperature')}. Using default 0.7.")
        merged_config["temperature"] = 0.7
    # Validate max_tokens
    mt = merged_config.get("max_tokens")
    if mt is not None:
        try:
            mt = int(mt)
            if mt <= 0:
                raise ValueError
            merged_config["max_tokens"] = mt
        except Exception:
            logger.warning(f"Invalid max_tokens: {mt}. Using unlimited.")
            merged_config["max_tokens"] = None
    # Validate n
    try:
        n = int(merged_config.get("n", 1))
    except Exception:
        logger.warning(f"Invalid n value: {merged_config.get('n')}. Using 1.")
        n = 1
    n = max(1, min(n, 5))
    merged_config["n"] = n
    
    try:
        # Handle streaming response
        if merged_config["stream"]:
            return _stream_mock_response(prompt, model_name, merged_config)
        
        response = _generate_mock_response(prompt, model_name, merged_config)
        # Handle multiple completions
        if n > 1:
            return [f"{response} [Completion {i+1}/{n}]".strip() for i in range(n)]
        return response
        
    except Exception as e:
        logger.error(f"Error in mock response generation: {str(e)}", exc_info=True)
        raise MockError(str(e)) from e
