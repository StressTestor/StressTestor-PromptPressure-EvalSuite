"""
PromptPressure Groq Adapter v1.5.2

Handles API calls to Groq for LLM completion with improved error handling and retries.
"""

import os
import time
import json
import logging
import random
from typing import Optional, Dict, Any, Union, Iterator, List, Tuple
import requests
from requests.exceptions import RequestException, Timeout, HTTPError, ChunkedEncodingError

# Custom exceptions
class GroqError(Exception):
    """Base exception for Groq adapter errors."""
    pass

class GroqAPIError(GroqError):
    """Raised when the Groq API returns an error response."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(f"Groq API Error ({status_code}): {message}" if status_code else f"Groq API Error: {message}")

class GroqValidationError(GroqError):
    """Raised when input validation fails."""
    pass

class GroqRateLimitError(GroqAPIError):
    """Raised when rate limit is exceeded."""
    pass

# Set up logger
logger = logging.getLogger(__name__)

def _validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize the request payload.
    
    Args:
        payload: The request payload to validate
        
    Returns:
        Validated and normalized payload
        
    Raises:
        GroqValidationError: If the payload is invalid
    """
    validated = payload.copy()
    
    # Handle stop sequences (convert string to list if needed)
    if 'stop' in validated:
        if isinstance(validated['stop'], str):
            validated['stop'] = [validated['stop']]
        elif not (isinstance(validated['stop'], (list, tuple)) and 
                 all(isinstance(x, str) for x in validated['stop'])):
            raise GroqValidationError("stop must be a string or list of strings")
    
    # Validate temperature
    if 'temperature' in validated:
        try:
            temp = float(validated['temperature'])
            if not 0.0 <= temp <= 2.0:
                logger.warning(f"Temperature {temp} clamped to [0.0, 2.0]")
                validated['temperature'] = max(0.0, min(2.0, temp))
        except (TypeError, ValueError) as e:
            raise GroqValidationError(f"Invalid temperature: {validated['temperature']}") from e
    
    # Validate max_tokens
    if 'max_tokens' in validated:
        try:
            max_tokens = int(validated['max_tokens'])
            if max_tokens <= 0 or max_tokens > 8192:
                raise GroqValidationError(f"max_tokens must be between 1 and 8192, got {max_tokens}")
            validated['max_tokens'] = max_tokens
        except (TypeError, ValueError) as e:
            raise GroqValidationError(f"Invalid max_tokens: {validated['max_tokens']}") from e
    
    return validated

def _make_api_request(endpoint: str, headers: Dict[str, str], data: Dict[str, Any], 
                     timeout: int = 60, max_retries: int = 3) -> Dict[str, Any]:
    """
    Make an API request with retry logic and error handling.
    
    Implements exponential backoff with jitter and respects Retry-After headers.
    
    Args:
        endpoint: API endpoint URL
        headers: Request headers
        data: Request payload
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        
    Returns:
        Parsed JSON response
        
    Raises:
        GroqAPIError: For API-level errors
        GroqRateLimitError: For rate limiting errors
        RequestException: For network or other request errors
    """
    last_exception: Optional[Exception] = None
    last_status: Optional[int] = None
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Sending request to {endpoint}")
            response = requests.post(
                endpoint,
                headers=headers,
                json=data,
                timeout=timeout
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After')
                wait_time = min(60, (2 ** attempt))  # Cap at 60 seconds
                
                if retry_after:
                    try:
                        wait_time = float(retry_after) + random.uniform(0, 2.0)  # Add jitter
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
                
                raise GroqRateLimitError(
                    f"Rate limit exceeded after {max_retries} attempts",
                    status_code=429
                )
                
            response.raise_for_status()
            return response.json()
            
        except HTTPError as e:
            last_exception = e
            last_status = e.response.status_code if hasattr(e, 'response') else None
            
            if last_status and 400 <= last_status < 500:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get('error', {}).get('message', str(e))
                    raise GroqAPIError(
                        f"API error: {error_msg}",
                        status_code=last_status
                    ) from e
                except ValueError:
                    raise GroqAPIError(
                        f"HTTP {last_status} error: {str(e)}",
                        status_code=last_status
                    ) from e
            
            # For server errors (5xx), we'll retry
            wait_time = min(60, (2 ** attempt)) + random.uniform(0, 2.0)
            logger.warning(
                f"HTTP {last_status or 'unknown'} error on attempt {attempt + 1}/{max_retries}. "
                f"Retrying in {wait_time:.1f}s..."
            )
            
        except (RequestException, Timeout) as e:
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
    
    if isinstance(last_exception, HTTPError) and last_status == 429:
        raise GroqRateLimitError("Rate limit exceeded", status_code=429) from last_exception
    elif isinstance(last_exception, HTTPError) and last_status:
        raise GroqAPIError("API request failed", status_code=last_status) from last_exception
    else:
        raise RequestException(error_msg) from last_exception

def _build_request_payload(
    prompt: str,
    model_name: str,
    config: Dict[str, Any],
    stream: bool = False
) -> Tuple[Dict[str, Any], Dict[str, str], str]:
    """
    Build and validate the request payload for the Groq API.
    
    Args:
        prompt: The user's prompt text
        model_name: The model to use
        config: Configuration dictionary
        stream: Whether to stream the response
        
    Returns:
        Tuple of (payload, headers, endpoint)
        
    Raises:
        GroqValidationError: If the payload is invalid
    """
    # Validate API key
    api_key = os.getenv("GROQ_API_KEY") or config.get("groq_api_key")
    if not api_key:
        raise GroqValidationError("Missing GROQ_API_KEY in environment or config")
    
    # Build headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Build base payload
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.get("temperature", 0.7),
        "max_tokens": min(int(config.get("max_tokens", 1024)), 8192),
        "stream": stream,
        "n": max(1, min(int(config.get("n", 1)), 5))  # Allow 1-5 completions
    }
    
    # Add optional parameters
    optional_params = [
        "top_p", "frequency_penalty", "presence_penalty", 
        "stop", "logprobs", "echo", "logit_bias"
    ]
    for param in optional_params:
        if param in config:
            payload[param] = config[param]
    
    # Validate and normalize the payload
    validated_payload = _validate_payload(payload)
    
    # Get endpoint
    endpoint = config.get("groq_endpoint", "https://api.groq.com/openai/v1/chat/completions")
    
    return validated_payload, headers, endpoint

def generate_response(
    prompt: str, 
    model_name: str = "llama3-70b-8192", 
    config: Optional[Dict[str, Any]] = None,
    stream: bool = False
) -> Union[str, Iterator[str]]:
    """
    Generate a response from Groq LLM with enhanced error handling and retries.
    
    Args:
        prompt: User prompt text. Must be a non-empty string.
        model_name: Groq model name to use for generation.
        config: Optional configuration dictionary. May contain:
            - groq_api_key: API key (if not in environment)
            - temperature: Sampling temperature (0.0 to 2.0, default: 0.7)
            - max_tokens: Maximum tokens to generate (1-8192, default: 1024)
            - n: Number of completions to generate (1-5, default: 1)
            - top_p: Nucleus sampling parameter (0.0 to 1.0)
            - frequency_penalty: Penalty for frequent tokens (-2.0 to 2.0)
            - presence_penalty: Penalty for new tokens (-2.0 to 2.0)
            - stop: String or list of strings where the API will stop generating
            - request_timeout: Request timeout in seconds (default: 60)
            - max_retries: Maximum retry attempts (default: 3)
            - logprobs: Whether to return log probabilities
            - echo: Whether to echo the prompt in the response
            - logit_bias: Dictionary of token biases
        stream: If True, returns an iterator of response chunks.
        
    Returns:
        If stream=False (default): Generated response as a string (or list of strings if n>1)
        If stream=True: Iterator of response chunks
        
    Raises:
        GroqValidationError: For invalid inputs or configuration
        GroqAPIError: For API-level errors
        GroqRateLimitError: For rate limiting errors
        RequestException: For network or other request errors
    """
    if not prompt or not isinstance(prompt, str):
        raise GroqValidationError("Prompt must be a non-empty string")
    
    config = config or {}
    
    try:
        # Build and validate the request payload
        payload, headers, endpoint = _build_request_payload(
            prompt=prompt,
            model_name=model_name,
            config=config,
            stream=stream
        )
        
        logger.debug(f"Sending request to {endpoint} with model {model_name}")
        
        if stream:
            return _stream_response(
                endpoint=endpoint,
                headers=headers,
                data=payload,
                timeout=config.get("request_timeout", 60),
                max_retries=config.get("max_retries", 3)
            )
        
        # Handle non-streaming response
        response = _make_api_request(
            endpoint=endpoint,
            headers=headers,
            data=payload,
            timeout=config.get("request_timeout", 60),
            max_retries=config.get("max_retries", 3)
        )
        
        # Process the response
        choices = response.get("choices", [])
        if not choices:
            logger.warning("Received empty choices array in response")
            return ""
        
        # Handle multiple completions if n > 1
        n_completions = len(choices)
        if n_completions > 1:
            return [
                choice.get("message", {}).get("content", "")
                for choice in choices
            ]
        
        # Single completion
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            logger.warning("Received empty content in response")
        
        return content
        
    except json.JSONDecodeError as e:
        error_msg = f"Failed to decode API response: {str(e)}"
        logger.error(error_msg)
        raise GroqAPIError(error_msg) from e
    except GroqError:
        raise  # Re-raise our custom errors
    except Exception as e:
        error_msg = f"Unexpected error in generate_response: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise GroqError(error_msg) from e

def _stream_response(
    endpoint: str,
    headers: Dict[str, str],
    data: Dict[str, Any],
    timeout: int = 60,
    max_retries: int = 3
) -> Iterator[str]:
    """
    Handle streaming response from the API.
    
    Args:
        endpoint: API endpoint URL
        headers: Request headers
        data: Request payload
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        
    Yields:
        Response chunks as they arrive
        
    Raises:
        GroqAPIError: For API-level errors
        GroqRateLimitError: For rate limiting errors
        RequestException: For network or other request errors
    """
    last_exception: Optional[Exception] = None
    
    for attempt in range(max_retries):
        try:
            # Enable streaming in the request data
            stream_data = data.copy()
            stream_data["stream"] = True
            
            logger.debug(f"Starting streaming request to {endpoint}")
            
            with requests.post(
                endpoint,
                headers=headers,
                json=stream_data,
                stream=True,
                timeout=timeout
            ) as response:
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
                    
                    raise GroqRateLimitError(
                        f"Rate limit exceeded after {max_retries} attempts",
                        status_code=429
                    )
                
                response.raise_for_status()
                
                # Process the streaming response
                for line in response.iter_lines():
                    if not line:
                        continue
                        
                    line = line.decode('utf-8')
                    if not line.startswith('data: '):
                        continue
                        
                    chunk = line[6:]  # Remove 'data: ' prefix
                    if chunk == '[DONE]':
                        logger.debug("Received end of stream")
                        return
                        
                    try:
                        data = json.loads(chunk)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to decode chunk: {chunk[:200]}...")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing chunk: {str(e)}")
                        continue
                
                # If we get here, the stream completed successfully
                return
                
        except HTTPError as e:
            last_exception = e
            status_code = e.response.status_code if hasattr(e, 'response') else None
            
            if status_code and 400 <= status_code < 500:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get('error', {}).get('message', str(e))
                    raise GroqAPIError(
                        f"API error: {error_msg}",
                        status_code=status_code
                    ) from e
                except ValueError:
                    raise GroqAPIError(
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
                f"Stream error on attempt {attempt + 1}/{max_retries}. "
                f"Error: {str(e)[:200]}. Retrying in {wait_time:.1f}s..."
            )
        
        if attempt < max_retries - 1:  # Don't sleep on the last attempt
            time.sleep(wait_time)
    
    # If we get here, all retries failed
    error_msg = f"Stream failed after {max_retries} attempts. Last error: {str(last_exception)}"
    logger.error(error_msg)
    
    if isinstance(last_exception, HTTPError) and hasattr(last_exception, 'response'):
        status_code = last_exception.response.status_code
        if status_code == 429:
            raise GroqRateLimitError("Rate limit exceeded", status_code=429) from last_exception
        raise GroqAPIError("API request failed", status_code=status_code) from last_exception
    
    raise RequestException(error_msg) from last_exception
