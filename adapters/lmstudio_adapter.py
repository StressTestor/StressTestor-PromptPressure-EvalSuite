"""
LM Studio adapter for PromptPressure Eval Suite v1.5.2

Handles API calls to LM Studio's local inference server with improved error handling and retries.
"""

import os
import time
import json
import logging
import random
from typing import Optional, Dict, Any, Union, List, TypeVar, Iterator, Tuple, Generator
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
    "lmstudio_endpoint": "http://localhost:1234/v1/chat/completions"
}

# Set up logger
logger = logging.getLogger(__name__)

# Custom exceptions
class LMStudioError(Exception):
    """Base exception for LM Studio adapter errors."""
    pass

class LMStudioAPIError(LMStudioError):
    """Raised when the LM Studio API returns an error response."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(f"LM Studio API Error ({status_code}): {message}" if status_code 
                        else f"LM Studio API Error: {message}")

class LMStudioValidationError(LMStudioError):
    """Raised when input validation fails."""
    pass

class LMStudioRateLimitError(LMStudioAPIError):
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
        LMStudioValidationError: If the configuration is invalid
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
            raise LMStudioValidationError(f"Invalid temperature: {validated['temperature']}") from e
    
    # Validate max_tokens
    if 'max_tokens' in validated:
        try:
            max_tokens = int(validated['max_tokens'])
            if max_tokens <= 0:
                raise LMStudioValidationError(f"max_tokens must be positive, got {max_tokens}")
            validated['max_tokens'] = max_tokens
        except (TypeError, ValueError) as e:
            raise LMStudioValidationError(f"Invalid max_tokens: {validated['max_tokens']}") from e
    
    # Validate request_timeout
    if 'request_timeout' in validated:
        try:
            timeout = int(validated['request_timeout'])
            if timeout <= 0:
                raise LMStudioValidationError(f"request_timeout must be positive, got {timeout}")
            validated['request_timeout'] = timeout
        except (TypeError, ValueError) as e:
            raise LMStudioValidationError(f"Invalid request_timeout: {validated['request_timeout']}") from e
    
    # Validate max_retries
    if 'max_retries' in validated:
        try:
            max_retries = int(validated['max_retries'])
            if max_retries < 0:
                raise LMStudioValidationError(f"max_retries must be non-negative, got {max_retries}")
            validated['max_retries'] = max_retries
        except (TypeError, ValueError) as e:
            raise LMStudioValidationError(f"Invalid max_retries: {validated['max_retries']}") from e
    
    return validated

def _stream_response(
    endpoint: str,
    payload: Dict[str, Any],
    config: Dict[str, Any]
) -> Generator[Tuple[Optional[str], bool], None, None]:
    """
    Handle streaming response from the API.
    
    Args:
        endpoint: API endpoint URL
        payload: Request payload
        config: Configuration dictionary
        
    Yields:
        Tuple of (chunk, is_done) where chunk is the response text or None if done,
        and is_done is a boolean indicating if the stream is complete
        
    Raises:
        LMStudioAPIError: For API-level errors
        RequestException: For network or other request errors
    """
    # Enable streaming in the payload
    stream_payload = {**payload, "stream": True}
    
    try:
        with requests.post(
            endpoint,
            json=stream_payload,
            stream=True,
            timeout=config.get('request_timeout', DEFAULT_CONFIG['request_timeout'])
        ) as response:
            response.raise_for_status()
            
            buffer = ""
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
            
            # Signal completion with None and any remaining buffer
            if buffer:
                yield buffer, True
            
    except Exception as e:
        logger.error(f"Error in streaming response: {str(e)}")
        raise LMStudioAPIError(f"Streaming error: {str(e)}") from e
    finally:
        # Always yield None at the end to signal completion
        yield None, True

def _make_api_request(
    endpoint: str,
    payload: Dict[str, Any],
    config: Dict[str, Any]
) -> Union[Dict[str, Any], Generator[Tuple[Optional[str], bool], None, None]]:
    """
    Make an API request with retry logic and error handling.
    
    Args:
        endpoint: API endpoint URL
        payload: Request payload
        config: Configuration dictionary with 'stream' and other options
        
    Returns:
        For non-streaming: Parsed JSON response as dict
        For streaming: Generator of (chunk, is_done) tuples
        
    Raises:
        LMStudioAPIError: For API-level errors
        LMStudioRateLimitError: For rate limiting errors
        RequestException: For network or other request errors
    """
    last_exception = None
    max_retries = config.get('max_retries', DEFAULT_CONFIG['max_retries'])
    is_streaming = payload.get('stream', False)
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Sending request to {endpoint} (streaming: {is_streaming})")
            
            if is_streaming:
                return _stream_response(endpoint, payload, config)
                
            # Non-streaming request
            response = requests.post(
                endpoint,
                json=payload,
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
                
                raise LMStudioRateLimitError(
                    f"Rate limit exceeded after {max_retries} attempts",
                    status_code=429
                )
            
            response.raise_for_status()
            return response.json()
            
        except HTTPError as e:
            last_exception = e
            status_code = e.response.status_code if hasattr(e, 'response') else None
            
            if status_code and 400 <= status_code < 500:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get('error', {}).get('message', str(e))
                    raise LMStudioAPIError(
                        f"API error: {error_msg}",
                        status_code=status_code
                    ) from e
                except ValueError:
                    raise LMStudioAPIError(
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
            raise LMStudioRateLimitError("Rate limit exceeded", status_code=429) from last_exception
        raise LMStudioAPIError("API request failed", status_code=status_code) from last_exception
    
    raise RequestException(error_msg) from last_exception

def generate_response(
    prompt: str, 
    model_name: str, 
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a response from LM Studio's local inference server.
    
    Args:
        prompt: User prompt text. Must be a non-empty string.
        model_name: Model identifier as configured in LM Studio.
        config: Optional configuration dictionary. May contain:
            - lmstudio_endpoint: API endpoint URL (default: http://localhost:1234/v1/chat/completions)
            - temperature: Sampling temperature (0.0 to 2.0, default: 0.7)
            - max_tokens: Maximum tokens to generate (default: 1024)
            - request_timeout: Request timeout in seconds (default: 60)
            - max_retries: Maximum retry attempts (default: 3)
            - top_p: Nucleus sampling parameter (0.0 to 1.0)
            - frequency_penalty: Penalty for frequent tokens (-2.0 to 2.0)
            - presence_penalty: Penalty for new tokens (-2.0 to 2.0)
            - stop: String or list of strings where the API will stop generating
            
    Returns:
        Model-generated response as a string
        
    Raises:
        LMStudioValidationError: For invalid inputs or configuration
        LMStudioAPIError: For API-level errors
        LMStudioRateLimitError: For rate limiting errors
        RequestException: For network or other request errors
    """
    if not prompt or not isinstance(prompt, str):
        raise LMStudioValidationError("Prompt must be a non-empty string")
    
    # Merge with default config and validate
    merged_config = {**DEFAULT_CONFIG, **(config or {})}
    try:
        merged_config = _validate_config(merged_config)
    except LMStudioValidationError as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        raise
    
    # Get endpoint from config or environment variable
    endpoint = os.getenv("LMSTUDIO_ENDPOINT", merged_config["lmstudio_endpoint"])
    
    # Prepare the request payload
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": merged_config["temperature"],
    }
    
    # Add optional parameters if they exist in the config
    optional_params = [
        "max_tokens", "top_p", "frequency_penalty", 
        "presence_penalty", "stop"
    ]
    for param in optional_params:
        if param in merged_config and merged_config[param] is not None:
            payload[param] = merged_config[param]
    
    try:
        # Make the API request
        response = _make_api_request(endpoint, payload, merged_config)
        
        # Handle streaming response
        if merged_config["stream"]:
            # For streaming, return an iterator that yields chunks
            def stream_generator() -> Iterator[str]:
                for chunk, is_done in response:
                    if chunk is not None:
                        yield chunk
                yield None  # Signal completion with None
            
            return stream_generator()
        
        # Handle non-streaming response
        try:
            choices = response.get("choices", [])
            if not choices:
                raise LMStudioAPIError("No choices in response")
                
            # For n>1, return a list of responses
            if len(choices) > 1:
                return [
                    choice.get("message", {}).get("content", "").strip()
                    for choice in choices
                    if choice.get("message", {}).get("content")
                ]
            
            # For n=1, return a single string
            return choices[0].get("message", {}).get("content", "").strip()
            
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            error_msg = f"Malformed LM Studio response: {response}"
            logger.error(error_msg)
            raise LMStudioAPIError(error_msg) from e
            
    except json.JSONDecodeError as e:
        error_msg = f"Failed to decode API response: {str(e)}"
        logger.error(error_msg)
        raise LMStudioAPIError(error_msg) from e
    except LMStudioError:
        raise  # Re-raise our custom errors
    except Exception as e:
        error_msg = f"Unexpected error in generate_response: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise LMStudioError(error_msg) from e
