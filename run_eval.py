#!/usr/bin/env python3
#
# run_eval.py
#
# PromptPressure Eval Suite - Runner Script
#
# Version: 1.5.0-patch8
# Generated: 2025-06-14
# Updated: 2025-06-14

"""
PromptPressure Eval Runner v1.5.0-patch8
----------------------------------------

A robust evaluation suite for LLM models with pluggable adapters, YAML configuration,
and comprehensive result analysis. Handles dataset validation, model inference,
and post-processing with detailed metrics collection.

Features:
- Support for multiple model providers (Groq, LM Studio, OpenAI, etc.)
- Configurable via YAML and environment variables
- Detailed metrics collection and analysis
- Automatic dataset validation and fallback
- Comprehensive error handling and logging

Example usage:
    python run_eval.py --model groq --config config.yaml --dataset evals_dataset.json
"""

import os
import sys
import json
import time
import logging
import argparse
import pandas as pd
import yaml
import subprocess
import glob
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dotenv import load_dotenv

# Constants
DEFAULT_CONFIG = {
    "model": "mock",
    "model_name": "qwen/qwen3-8b",
    "is_simulation": True,
    "dataset": "evals_dataset.json",
    "output_dir": "outputs",
    "max_retries": 3,
    "request_timeout": 60,  # seconds
}

# Configure logging
def setup_logging(output_dir: str = 'outputs') -> None:
    """Set up logging configuration with file output in the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'promptpressure.log')
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )

# Logger will be configured in main()
logger = logging.getLogger(__name__)

# Common refusal phrases
REFUSAL_PHRASES = [
    "i'm sorry", "i cannot", "as an ai", "i am unable", "it would be inappropriate",
    "i cannot comply", "i don't think", "i apologize", "not allowed", "it's not possible"
]

def setup_environment(output_dir: str = 'outputs') -> None:
    """
    Initialize environment variables and validate required settings.
    
    Args:
        output_dir: Directory where logs and outputs will be stored
    """
    load_dotenv()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Reconfigure logging with the correct output directory
    setup_logging(output_dir)
    
    # Add any required environment variable validations here
    required_vars = []  # Add any required env vars here
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise EnvironmentError(error_msg)


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and validate YAML configuration from a file path.

    Args:
        path: Path to the YAML configuration file

    Returns:
        dict: Configuration dictionary with defaults for missing values
    """
    config = DEFAULT_CONFIG.copy()
    
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding='utf-8') as f:
                user_config = yaml.safe_load(f) or {}
                config.update(user_config)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            raise
    
    return config


def detect_refusal(text: str) -> bool:
    """
    Detect common refusal phrases in the text.

    Args:
        text: Input text to analyze

    Returns:
        bool: True if refusal is detected, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
    return any(phrase in text.lower() for phrase in REFUSAL_PHRASES)


def detect_format_bullet(text: str, min_items: int = 3) -> bool:
    """
    Detect if text is formatted as a bulleted list with at least min_items.

    Args:
        text: Input text to analyze
        min_items: Minimum number of bullet points required

    Returns:
        bool: True if text is a bulleted list with at least min_items
    """
    if not text or not isinstance(text, str):
        return False
        
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    bullet_chars = ('-', '*', '•', '‣', '⁃')
    bullet_lines = [line for line in lines if any(line.startswith(char) for char in bullet_chars)]
    return len(bullet_lines) >= min_items


def find_valid_dataset(default_path: str, max_size_mb: int = 10) -> Optional[str]:
    """
    Find a valid dataset JSON file, checking default and fallback locations.

    Args:
        default_path: Primary path to check first
        max_size_mb: Maximum file size in MB to consider

    Returns:
        str: Path to a valid dataset file, or None if none found
    """
    max_size = max_size_mb * 1024 * 1024  # Convert MB to bytes
    candidates = []

    def is_valid_json(path: str) -> bool:
        """Check if file is valid JSON and within size limits."""
        try:
            if os.path.getsize(path) > max_size:
                logger.warning(f"File too large (>{max_size_mb}MB): {path}")
                return False
            with open(path, 'r', encoding='utf-8') as f:
                json.load(f)
            return True
        except (json.JSONDecodeError, IOError) as e:
            logger.debug(f"Invalid JSON in {path}: {e}")
            return False

    # Check default path first
    if default_path and os.path.exists(default_path) and os.path.getsize(default_path) > 0:
        candidates.append(default_path)

    # Add other matching JSON files
    candidates.extend(
        path for path in glob.glob("*_dataset*.json")
        if path != default_path 
        and os.path.getsize(path) > 0
    )

    # Return first valid candidate
    for path in candidates:
        if is_valid_json(path):
            logger.info(f"Using dataset: {path}")
            return path

    logger.error("No valid dataset files found")
    return None


def run_adapter_with_retry(adapter_fn, prompt: str, model_name: str, config: Dict[str, Any], max_retries: int = 3) -> str:
    """Run adapter function with retry logic.
    
    Args:
        adapter_fn: Adapter function to call
        prompt: Input prompt text
        model_name: Name of the model to use
        config: Configuration dictionary
        max_retries: Maximum number of retry attempts
        
    Returns:
        str: Model response or error message
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return adapter_fn(prompt, model_name, config)
        except Exception as e:
            last_error = e
            wait_time = 2 ** attempt  # Exponential backoff
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                f"Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)
    
    logger.error(f"All {max_retries} attempts failed. Last error: {last_error}")
    raise last_error


def process_prompt(adapter_fn: callable, prompt_data: Dict[str, Any], model_key: str, 
                  model_name: str, is_simulation: bool, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single prompt and return results.
    
    Args:
        adapter_fn: The adapter function to use for processing
        prompt_data: Dictionary containing prompt data
        model_key: Key identifying the model
        model_name: Name of the model
        is_simulation: Whether to run in simulation mode
        config: Configuration dictionary
        
    Returns:
        Dictionary containing prompt processing results
    """
    prompt_text = prompt_data.get("input", "")
    prompt_id = prompt_data.get("id", hash(prompt_text) % 10000)  # Simple hash if no ID
    
    result = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "prompt_id": prompt_id,
        "category": prompt_data.get("category", ""),
        "model_provider": model_key,
        "model_name": model_name,
        "prompt_length_chars": len(prompt_text),
        "prompt_word_count": len(prompt_text.split()),
        "response_length_chars": 0,
        "response_word_count": 0,
        "input": prompt_text,
        "response": "",
        "refusal": None,
        "format_compliance": None,
        "response_time_sec": 0,
        "error_message": None,
        "is_simulation": is_simulation
    }
    
    try:
        start_time = time.time()
        response = run_adapter_with_retry(adapter_fn, prompt_text, model_name, config)
        end_time = time.time()
        
        if not isinstance(response, str):
            response = str(response)
            
        result.update({
            "response": response,
            "response_length_chars": len(response),
            "response_word_count": len(response.split()),
            "refusal": detect_refusal(response),
            "format_compliance": (
                detect_format_bullet(response)
                if "bullet" in prompt_text.lower()
                else None
            ),
            "response_time_sec": round(end_time - start_time, 2)
        })
        
        logger.info(f"Processed prompt {prompt_id} successfully")
        
    except Exception as e:
        error_msg = str(e)
        result.update({
            "response": f"ERROR: {error_msg}",
            "error_message": error_msg
        })
        logger.error(f"Error processing prompt {prompt_id}: {error_msg}", exc_info=True)
    
    return result


def run_eval(args):
    """
    Run the evaluation pipeline.
    
    Args:
        args: Command line arguments
    """
    start_time = time.time()
    
    # Load and validate configuration
    try:
        config = load_config(args.config)
        model_key = args.model or config.get("model")
        if not model_key:
            raise ValueError("Model key must be provided via --model or config file")
            
        model_name = config.get("model_name", model_key)
        is_simulation = args.simulation or config.get("is_simulation", True)
        dataset_path = find_valid_dataset(args.dataset or config.get("dataset"))
        
        if not dataset_path:
            raise FileNotFoundError(
                f"No valid JSON dataset found (tried {args.dataset or 'default'} and fallbacks)"
            )
            
        # Set up output directory
        output_dir = Path(config.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up output file
        if args.output:
            output_path = output_dir / args.output
        else:
            safe_name = f"{model_name.replace('/', '_').replace('-', '_')}_{int(time.time())}"
            output_path = output_dir / f"eval_results_{safe_name}.csv"
        
        logger.info(f"Starting evaluation with model: {model_name}")
        logger.info(f"Using dataset: {dataset_path}")
        logger.info(f"Output will be saved to: {output_path}")
        
        # List available adapters if requested
        if getattr(args, "list_adapters", False):
            from adapters import ADAPTER_REGISTRY
            print("\nAvailable adapters:")
            for key in sorted(ADAPTER_REGISTRY.keys()):
                print(f"  - {key}")
            return
            
        # Import adapters here to avoid circular imports
        from adapters import ADAPTER_REGISTRY
        
        # Validate adapter
        if model_key not in ADAPTER_REGISTRY:
            available = ", ".join(sorted(ADAPTER_REGISTRY.keys()))
            raise ValueError(
                f"Model adapter '{model_key}' not found. Available: {available}"
            )
            
        adapter_fn = ADAPTER_REGISTRY[model_key]
        
        # Load and validate dataset
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                prompts = json.load(f)
                if not isinstance(prompts, list):
                    raise ValueError("Dataset must be a JSON array of prompt objects")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
            
        # Process prompts
        results = []
        total_prompts = len(prompts)
        logger.info(f"Processing {total_prompts} prompts...")
        
        for i, prompt_data in enumerate(prompts, 1):
            if not isinstance(prompt_data, dict):
                logger.warning(f"Skipping invalid prompt at index {i-1}: not a dictionary")
                continue
                
            result = process_prompt(adapter_fn, prompt_data, model_key, model_name, is_simulation, config)
            results.append(result)
            
            # Log progress
            if i % 10 == 0 or i == total_prompts:
                logger.info(f"Progress: {i}/{total_prompts} prompts processed")
        
        # Save results
        if results:
            df = pd.DataFrame(results)
            try:
                df.to_csv(output_path, index=False, encoding='utf-8')
                logger.info(f"Results saved to {output_path}")
                
                # Run post-analysis if not in simulation mode
                if not is_simulation and not args.no_analysis:
                    run_post_analysis(output_dir, output_path)
                    
            except Exception as e:
                logger.error(f"Failed to save results: {e}", exc_info=True)
                raise
        else:
            logger.warning("No results to save")
            
    except Exception as e:
        logger.critical(f"Evaluation failed: {e}", exc_info=True)
        raise
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Evaluation completed in {elapsed:.2f} seconds")

def run_post_analysis(output_dir: Path, results_path: Path) -> None:
    """
    Run post-analysis on the results.
    
    Args:
        output_dir: Directory containing the results
        results_path: Path to the results CSV file
    """
    analysis_script = Path(__file__).parent / "deepseek_post_analysis.py"
    if not analysis_script.exists():
        logger.warning("Post-analysis script not found. Skipping...")
        return
        
    logger.info("Starting post-analysis...")
    try:
        original_dir = Path.cwd()
        os.chdir(output_dir)
        
        result = subprocess.run(
            [sys.executable, str(analysis_script)],
            check=False,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Post-analysis failed with code {result.returncode}")
            if result.stderr:
                logger.error(f"Error output: {result.stderr}")
        else:
            logger.info("Post-analysis completed successfully")
            if result.stdout:
                logger.debug(f"Analysis output: {result.stdout}")
                
    except Exception as e:
        logger.error(f"Error during post-analysis: {e}", exc_info=True)
    finally:
        os.chdir(original_dir)


def main() -> None:
    """Main entry point for the script."""
    # Initial basic logging setup (will be reconfigured after parsing args)
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up a temporary logger for pre-configuration messages
    temp_logger = logging.getLogger('promptpressure')
    temp_logger.setLevel(logging.WARNING)
    
    parser = argparse.ArgumentParser(
        description="PromptPressure Eval Runner - Comprehensive LLM Evaluation Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Target model key (e.g., groq, lmstudio, openai, mock)"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="Path to evaluation dataset JSON file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output CSV file name (placed in outputs/)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--simulation", "-s",
        action="store_true",
        help="Run in simulation mode (no actual API calls)"
    )
    
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip post-analysis step"
    )
    
    parser.add_argument(
        "--list-adapters",
        action="store_true",
        help="List available model adapters and exit"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (use -vv for debug)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store output files (default: 'outputs' or value from config)"
    )
    
    args = parser.parse_args()
    
    # Configure logging level based on verbosity
    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    
    # Load config and determine output directory
    config = load_config(args.config)
    output_dir = args.output_dir or config.get('output_dir', 'outputs')
    
    # Now that we have the output directory, set up proper logging
    setup_logging(output_dir)
    
    # Re-fetch the logger to ensure we're using the properly configured one
    global logger
    logger = logging.getLogger(__name__)
    
    # Setup environment with proper output directory
    setup_environment(output_dir)
    
    # Set log level for all loggers
    logging.getLogger().setLevel(log_level)
    
    # Log startup information
    logger.info("=" * 50)
    logger.info(f"Starting PromptPressure Eval Runner (v1.5.0-patch8)")
    logger.info(f"Log level: {logging.getLevelName(log_level)}")
    logger.info(f"Output directory: {os.path.abspath(output_dir)}")
    logger.info("=" * 50)
    
    try:
        run_eval(args)
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
