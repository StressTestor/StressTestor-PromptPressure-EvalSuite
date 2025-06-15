"""
PromptPressure OpenAI Adapter v1.5.2

Handles API calls to OpenAI's chat completions API with improved error handling and streaming support.
"""

import os
import time
import json
import logging
import random
from typing import Optional, Dict, Any, Union, List, Iterator, Tuple, Generator, TypeVar
import requests
from requests.exceptions import RequestException, Timeout, HTTPError, ChunkedEncodingError

# Type variable for generic return types
T = TypeVar('T')

# Default configuration
DEFAULT_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 1024,
    "request_timeout": 60,
    "max_retries": 3,
    "n": 1,
    "stream": False,
    "openai_endpoint": "https://api.openai.com/v1/chat/completions",
    "organization": None,
    "user": None,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop": None
}

# Set up logger
logger = logging.getLogger(__name__)

# Custom exceptions
class OpenAIError(Exception):
    """Base exception for OpenAI adapter errors."""
    pass

class OpenAIAPIError(OpenAIError):
    """Raised when the OpenAI API returns an error response."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(
            f"OpenAI API Error ({status_code}): {message}" if status_code 
            else f"OpenAI API Error: {message}"
        )

class OpenAIValidationError(OpenAIError):
    """Raised when input validation fails."""
    pass

class OpenAIRateLimitError(OpenAIAPIError):
    """Raised when rate limit is exceeded."""
    pass

def _validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize the configuration.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Validated and normalized configuration
        
    Raises:
        OpenAIValidationError: If the configuration is invalid
    """
    validated = config.copy()
    
    # Validate temperature
    if 'temperature' in validated:
        try:
            temp = float(validated['temperature'])
            if not 0.0 <= temp <= 2.0:
                logger.warning(f"Temperature {temp} clamped to [0.0, 2.0]")
                validated['temperature'] = max(0.0, min(2.0, temp))
        except (TypeError, ValueError) as e:
            raise OpenAIValidationError(f"Invalid temperature: {validated['temperature']}") from e
    
    # Validate max_tokens
    if 'max_tokens' in validated:
        try:
            max_tokens = int(validated['max_tokens'])
            if max_tokens <= 0:
                raise OpenAIValidationError(f"max_tokens must be positive, got {max_tokens}")
            validated['max_tokens'] = max_tokens
        except (TypeError, ValueError) as e:
            raise OpenAIValidationError(f"Invalid max_tokens: {validated['max_tokens']}") from e
    
    # Validate request_timeout
    if 'request_timeout' in validated:
        try:
            timeout = int(validated['request_timeout'])
            if timeout <= 0:
                raise OpenAIValidationError(f"request_timeout must be positive, got {timeout}")
            validated['request_timeout'] = timeout
        except (TypeError, ValueError) as e:
            raise OpenAIValidationError(f"Invalid request_timeout: {validated['request_timeout']}") from e
    
    # Validate max_retries
    if 'max_retries' in validated:
        try:
            max_retries = int(validated['max_retries'])
            if max_retries < 0:
                raise OpenAIValidationError(f"max_retries must be non-negative, got {max_retries}")
            validated['max_retries'] = max_retries
        except (TypeError, ValueError) as e:
            raise OpenAIValidationError(f"Invalid max_retries: {validated['max_retries']}") from e
    
    # Validate n (number of completions)
    if 'n' in validated:
        try:
            n = int(validated['n'])
            if n <= 0:
                raise OpenAIValidationError(f"n must be positive, got {n}")
            validated['n'] = n
        except (TypeError, ValueError) as e:
            raise OpenAIValidationError(f"Invalid n: {validated['n']}") from e
    
    return validated

def _stream_response(
    response: requests.Response
) -> Generator[Tuple[Optional[str], bool], None, None]:
    """
    Handle streaming response from the OpenAI API.
    
    Args:
        response: The streaming response object
        
    Yields:
        Tuple of (chunk, is_done) where chunk is the response text or None if done,
        and is_done is a boolean indicating if the stream is complete
        
    Raises:
        OpenAIAPIError: For API-level errors
        RequestException: For network or other request errors
    """
    buffer = ""
    try:
        for line in response.iter_lines():
            if not line:
                continue
                
            # Remove 'data: ' prefix and handle [DONE] message
            if line.startswith(b'data: '):
                line = line[6:]
            
            if line == b'[DONE]':
                break
                
            try:
                data = json.loads(line)
                chunk = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if chunk:
                    buffer += chunk
                    yield chunk, False
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode chunk: {line}")
        
        # Signal completion with any remaining buffer
        if buffer:
            yield buffer, True
            
    except Exception as e:
        logger.error(f"Error in streaming response: {str(e)}")
        raise OpenAIAPIError(f"Streaming error: {str(e)}") from e
    finally:
        # Always yield None at the end to signal completion
        yield None, True

def _make_api_request(
    endpoint: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    config: Dict[str, Any]
) -> Union[Dict[str, Any], Generator[Tuple[Optional[str], bool], None, None]]:
    """
    Make an API request with retry logic and error handling.
    
    Args:
        endpoint: API endpoint URL
        headers: Request headers including authorization
        payload: Request payload
        config: Configuration dictionary with 'stream' and other options
        
    Returns:
        For non-streaming: Parsed JSON response as dict
        For streaming: Generator of (chunk, is_done) tuples
        
    Raises:
        OpenAIAPIError: For API-level errors
        OpenAIRateLimitError: For rate limiting errors
        RequestException: For network or other request errors
    """
    last_exception = None
    max_retries = config.get('max_retries', DEFAULT_CONFIG['max_retries'])
    is_streaming = payload.get('stream', False)
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Sending request to {endpoint} (streaming: {is_streaming})")
            
            # Make the request with streaming if enabled
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                stream=is_streaming,
                timeout=config.get('request_timeout', DEFAULT_CONFIG['request_timeout'])
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After')
                wait_time = min(60, (2 ** attempt))  # Cap at 60 seconds
                
                if retry_after:
                    try:
                        wait_time = float(retry_after) + random.uniform(0, 2.0)
                        logger.warning(
                            f"Rate limited (429). Server requested wait: {retry_after}s. "
                            f"Waiting {wait_time:.1f}s..."
                        )
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Rate limited (429). Invalid Retry-After: {retry_after}. "
                            f"Using exponential backoff: {wait_time:.1f}s"
                        )
                
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                
                raise OpenAIRateLimitError(
                    f"Rate limit exceeded after {max_retries} attempts",
                    status_code=429
                )
            
            response.raise_for_status()
            
            # Handle streaming response
            if is_streaming:
                return _stream_response(response)
                
            # Return parsed JSON for non-streaming
            return response.json()
            
        except HTTPError as e:
            last_exception = e
            status_code = e.response.status_code if hasattr(e, 'response') else None
            
            if status_code and 400 <= status_code < 500:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get('error', {}).get('message', str(e))
                    raise OpenAIAPIError(
                        f"API error: {error_msg}",
                        status_code=status_code
                    ) from e
                except ValueError:
                    raise OpenAIAPIError(
                        f"HTTP {status_code} error: {str(e)}",
                        status_code=status_code
                    ) from e
            
            # For server errors (5xx), we'll retry
            wait_time = min(60, (2 ** attempt)) + random.uniform(0, 2.0)
            logger.warning(
                f"HTTP {status_code or 'unknown'} error on attempt {attempt + 1}/{max_retries}. "
                f"Retrying in {wait_time:.1f}s..."
            )
            
        except (RequestException, Timeout, ChunkedEncodingError) as e:
            last_exception = e
            wait_time = min(60, (2 ** attempt)) + random.uniform(0, 2.0)
            logger.warning(
                f"Request failed on attempt {attempt + 1}/{max_retries}. "
                f"Error: {str(e)[:200]}. Retrying in {wait_time:.1f}s..."
            )
        
        if attempt < max_retries - 1:  # Don't sleep on the last attempt
            time.sleep(wait_time)
    
    # If we get here, all retries failed
    error_msg = f"All {max_retries} attempts failed. Last error: {str(last_exception)}"
    logger.error(error_msg)
    
    if isinstance(last_exception, HTTPError) and hasattr(last_exception, 'response'):
        status_code = last_exception.response.status_code
        if status_code == 429:
            raise OpenAIRateLimitError("Rate limit exceeded", status_code=429) from last_exception
        raise OpenAIAPIError("API request failed", status_code=status_code) from last_exception
    
    raise RequestException(error_msg) from last_exception

def generate_response(
    prompt: str,
    model_name: str = "gpt-4-1106-preview",
    config: Optional[Dict[str, Any]] = None
) -> Union[str, List[str], Iterator[str]]:
    """
    Generate a response from OpenAI's chat completions API.
    
    Args:
        prompt: User prompt text. Must be a non-empty string.
        model_name: Model identifier (e.g., "gpt-4-1106-preview").
        config: Optional configuration dictionary. May contain:
            - openai_api_key: API key (required if not in OPENAI_API_KEY env var)
            - openai_endpoint: API endpoint URL
            - organization: Organization ID for usage tracking
            - temperature: Sampling temperature (0.0 to 2.0, default: 0.7)
            - max_tokens: Maximum tokens to generate (default: 1024)
            - n: Number of completions to generate (default: 1)
            - stream: Whether to stream the response (default: False)
            - request_timeout: Request timeout in seconds (default: 60)
            - max_retries: Maximum retry attempts (default: 3)
            - top_p: Nucleus sampling parameter (0.0 to 1.0)
            - frequency_penalty: Penalty for frequent tokens (-2.0 to 2.0)
            - presence_penalty: Penalty for new tokens (-2.0 to 2.0)
            - stop: String or list of strings where the API will stop generating
            - user: A unique identifier for end-user tracking
            - logprobs: Include log probabilities on the logprobs most likely tokens
            - top_logprobs: Number of highest probability tokens to return at each position
            - echo: Echo back the prompt in addition to the completion
            - logit_bias: Modify the likelihood of specified tokens appearing in the completion
            
    Returns:
        - If stream=False and n=1: A single string response
        - If stream=False and n>1: A list of string responses
        - If stream=True: An iterator that yields response chunks
        
    Raises:
        OpenAIValidationError: For invalid inputs or configuration
        OpenAIAPIError: For API-level errors
        OpenAIRateLimitError: For rate limiting errors
        RequestException: For network or other request errors
    """
    if not prompt or not isinstance(prompt, str):
        raise OpenAIValidationError("Prompt must be a non-empty string")
    
    # Merge with default config and validate
    merged_config = {**DEFAULT_CONFIG, **(config or {})}
    try:
        merged_config = _validate_config(merged_config)
    except OpenAIValidationError as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        raise
    
    # Get API key with proper precedence: config > environment variable
    api_key = merged_config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIValidationError(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
            "or provide openai_api_key in config."
        )
    
    # Get endpoint with proper precedence: config > default
    endpoint = merged_config.get("openai_endpoint") or DEFAULT_CONFIG["openai_endpoint"]
    
    # Prepare request headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Add organization header if provided
    if merged_config.get("organization"):
        headers["OpenAI-Organization"] = merged_config["organization"]
    
    # Prepare the request payload
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": merged_config["temperature"],
        "n": merged_config["n"],
        "stream": merged_config["stream"]
    }
    
    # Add optional parameters if they exist in the config
    optional_params = [
        "max_tokens", "top_p", "frequency_penalty", 
        "presence_penalty", "stop", "user"
    ]
    for param in optional_params:
        if param in merged_config and merged_config[param] is not None:
            payload[param] = merged_config[param]
    
    try:
        # Make the API request
        response = _make_api_request(endpoint, headers, payload, merged_config)
        
        # Handle streaming response
        if merged_config["stream"]:
            # For streaming, return an iterator that yields chunks
            # Note: The iterator will end naturally when the stream is complete
            # without yielding a sentinel value, as this is the more common
            # pattern in Python iterators
            return (chunk for chunk, is_done in response if chunk is not None)
        
        # Handle non-streaming response
        choices = response.get("choices", [])
        if not choices:
            raise OpenAIAPIError("No choices in response")
            
        # For n>1, return a list of responses
        if len(choices) > 1:
            return [
                choice.get("message", {}).get("content", "").strip()
                for choice in choices
                if choice.get("message", {}).get("content")
            ]
        
        # For n=1, return a single string
        return choices[0].get("message", {}).get("content", "").strip()
        
    except json.JSONDecodeError as e:
        error_msg = f"Failed to decode API response: {str(e)}"
        logger.error(error_msg)
        raise OpenAIAPIError(error_msg) from e
    except OpenAIError:
        raise  # Re-raise our custom errors
    except Exception as e:
        error_msg = f"Unexpected error in generate_response: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise OpenAIError(error_msg) from e
