#!/usr/bin/env python3
# deepseek_post_analysis.py
# PromptPressure Post-Analysis Script
# Version: 1.5.2
# Generated: 2025-06-14
# Notable: Aggregates all CSVs in outputs/ and runs analysis using appropriate adapters

import os
import sys
import glob
import logging
from datetime import datetime
from dotenv import load_dotenv
from typing import Any, Dict, List, Union, Optional, Callable

# Import all available adapters
try:
    from adapters.openai_adapter import generate_response as openai_resp
except ImportError:
    openai_resp = None

try:
    from adapters.groq_adapter import generate_response as groq_resp
except ImportError:
    groq_resp = None

try:
    from adapters.mock_adapter import generate_response as mock_resp
except ImportError:
    mock_resp = None

try:
    from adapters.lmstudio_adapter import generate_response as lmstudio_resp
except ImportError:
    lmstudio_resp = None

import pandas as pd

# Configure logging with file and console output
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler('deepseek_analysis.log')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Adapter mapping with fallback to mock if available
ADAPTER_MAP = {
    'openai': openai_resp,
    'groq': groq_resp,
    'mock': mock_resp,
    'lmstudio': lmstudio_resp
}

# Remove None values from adapter map
ADAPTER_MAP = {k: v for k, v in ADAPTER_MAP.items() if v is not None}

if not ADAPTER_MAP:
    logger.error("No adapters available. Please install at least one adapter.")
    sys.exit(1)

# Default adapter (first available)
DEFAULT_ADAPTER = next(iter(ADAPTER_MAP.values()))

logger.info(f"Available adapters: {', '.join(ADAPTER_MAP.keys())}")

# Load environment variables and set up error log
load_dotenv()
ERROR_LOG_FILE = os.getenv("ERROR_LOG", "error_log")

# 1) Auto-discover any CSV files in this directory
pattern = "*.csv"
matches = glob.glob(pattern)
if not matches:
    message = f"[{datetime.utcnow().isoformat()}] No files matching '{pattern}' found in {os.getcwd()}"
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as logf:
        logf.write(message + "\n")
    print(message)
    sys.exit(1)

# 2) Read and aggregate all CSVs
combined_frames = []
for csv_file in matches:
    try:
        df = pd.read_csv(csv_file)
        combined_frames.append(df)
    except Exception as e:
        message = f"[{datetime.utcnow().isoformat()}] Failed to read '{csv_file}': {e}"
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as logf:
            logf.write(message + "\n")
        continue

def get_adapter_for_record(record: Dict[str, Any]) -> Callable:
    """Get the appropriate adapter function for the given record."""
    provider = record.get('model_provider', '').lower()
    return ADAPTER_MAP.get(provider, DEFAULT_ADAPTER)

def process_record(record: Dict[str, Any], idx: int, default_model: str) -> Optional[Dict[str, Any]]:
    """Process a single record with validation and error handling."""
    try:
        # Skip empty or invalid records
        if not record or not isinstance(record, dict):
            logger.warning(f"Skipping invalid record at index {idx}: Not a dictionary")
            return None
            
        # Extract prompt with validation
        prompt = record.get("input", "")
        if not isinstance(prompt, str) or not prompt.strip():
            logger.warning(f"Skipping empty prompt at index {idx}")
            return None
            
        # Get model name from record or use default
        model_name = record.get("model_name") or default_model
        
        # Get the appropriate adapter
        adapter = get_adapter_for_record(record)
        adapter_name = [k for k, v in ADAPTER_MAP.items() if v == adapter][0]
        logger.info(f"Processing record {idx} with {adapter_name} adapter")
        
        # Prepare config with safe defaults
        config = {
            "temperature": 0.2,
            "max_tokens": 2000,
            "request_timeout": 120,
            "max_retries": 3,
            **record.get("config", {})  # Allow record-specific config overrides
        }
        
        # Generate response with retry logic
        response = adapter(
            prompt=prompt,
            model_name=model_name,
            config=config
        )
        
        # Return successful analysis
        return {
            "index": idx,
            "provider": adapter_name,
            "model": model_name,
            "input": prompt,  # Store full prompt
            "input_preview": prompt[:200] + ("..." if len(prompt) > 200 else ""),
            "analysis": response if isinstance(response, str) else str(response),
            "status": "success"
        }
        
    except Exception as e:
        error_msg = f"Error processing record {idx}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "index": idx,
            "provider": adapter_name if 'adapter_name' in locals() else 'unknown',
            "model": model_name if 'model_name' in locals() else 'unknown',
            "input": str(prompt) if 'prompt' in locals() else '',
            "input_preview": str(prompt)[:200] + ("..." if len(str(prompt)) > 200 else ""),
            "analysis": f"Error: {str(e)}",
            "status": "error"
        }

# 3) Process records with parallel execution (if needed)
combined_df = pd.concat(combined_frames, ignore_index=True)

# Get default model name
if "model_name" in combined_df.columns and not combined_df["model_name"].empty:
    default_model = combined_df["model_name"].iloc[0]
else:
    default_model = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-coder")

# Process each record
analysis_results = []
for idx, record in enumerate(combined_df.to_dict(orient="records")):
    result = process_record(record, idx, default_model)
    if result:
        analysis_results.append(result)

def generate_analysis_summary(analysis_results: List[Dict[str, Any]]) -> str:
    """Generate a detailed analysis summary."""
    if not analysis_results:
        return "No valid prompts were processed successfully"
    
    analysis_df = pd.DataFrame(analysis_results)
    success_count = len(analysis_df[analysis_df["status"] == "success"])
    error_count = len(analysis_df[analysis_df["status"] == "error"])
    total = len(analysis_results)
    
    # Group by provider and model
    provider_stats = analysis_df.groupby(['provider', 'model'])['status'] \
        .value_counts() \
        .unstack(fill_value=0) \
        .to_string()
    
    # Get sample analyses (up to 3)
    samples = analysis_df[analysis_df['status'] == 'success'].head(3)
    sample_analyses = "\n".join(
        f"{i+1}. Model: {row['model']} (Provider: {row['provider']})\n   Input: {row['input_preview']}"
        for i, (_, row) in enumerate(samples.iterrows())
    )
    
    # Build summary without indentation issues
    summary_parts = [
        "Deepseek Analysis Summary",
        "=" * 50,
        f"Total records processed: {total}",
        f"Successful analyses: {success_count} ({success_count/total*100:.1f}%)",
        f"Failed analyses: {error_count} ({error_count/total*100:.1f}%)",
        "",
        "Breakdown by Provider/Model:",
        provider_stats,
        "",
        "Sample of successful analyses:",
        sample_analyses if not samples.empty else "No successful analyses to display"
    ]
    
    return "\n".join(summary_parts)

# Generate final analysis
try:
    analysis = generate_analysis_summary(analysis_results)
    logger.info("Analysis completed successfully")
    
except Exception as e:
    message = f"[{datetime.utcnow().isoformat()}] Failed to generate analysis summary: {e}"
    logger.error(message, exc_info=True)
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as logf:
        logf.write(message + "\n")
    sys.exit(1)

# 4) Save analysis result to disk
out_path = "deepseek_post_analysis.txt"
try:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(analysis)
    print(f"[✓] Deepseek analysis saved to '{out_path}'")
except Exception as e:
    message = f"[{datetime.utcnow().isoformat()}] Failed to write analysis file: {e}"
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as logf:
        logf.write(message + "\n")
    print(message)
    sys.exit(1)
