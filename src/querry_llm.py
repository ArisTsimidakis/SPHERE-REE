import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Dict, Any, Optional
from helpers import check_bounds
import pandas as pd
import yaml
from openai import OpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)
config = None 

# Specify directories for reference
script_dir = Path(__file__).parent.resolve()
root_dir = script_dir.parent


@dataclass
class Config:
    # IO files and directories
    queries_file = os.path.join(root_dir, "experiment", "results", "llm_queries.csv")
    answers_file = os.path.join(root_dir, "experiment", "results", "llm_chatgpt_answers.csv")
    failed_queries_file = os.path.join(root_dir, "experiment", "results", "failed_queries.csv")
    evaluation_file = os.path.join(root_dir, "experiment", "results", "llm_evaluation.csv")
    api_key_file = os.path.join(root_dir, "openai_api_key.txt")
    temp_dir = os.path.join(root_dir, "temp_snippets")
    
    # Logging
    log_file: Optional[str] = None
    log_level: Optional[str] = "INFO"
    
    # ChatGPT settings
    model_name: Optional[str] = "gpt-4o-mini"
    batch_size = 20
    max_workers: int = max(cpu_count(), 1)


def parse_args():
    parser = argparse.ArgumentParser (description="Query ChatGPT for Kubernetes YAML refactoring suggestions and evaluate results.")

    # Processing 
    parser.add_argument("--start-index", type = int, default = 0, help = "Starting index for processing queries")
    parser.add_argument("--end-index", type = int, help = "Ending index for processing queries")
    
    # Logging 
    parser.add_argument("--log-file", help="Path to log file", default = None)
    parser.add_argument("--log-level", help="Logging level", default = "INFO", choices = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    
    # Model 
    parser.add_argument("--model", help="ChatGPT model to use", default="gpt-4o-mini")
    
    # Threading
    parser.add_argument("--max-workers", type=int, help="Maximum number of worker threads", default = max(cpu_count(), 1))
    
    args = parser.parse_args()
    return args

def setup_logging() -> None:
    """Configure logging handlers and format"""
    global config
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.handlers.clear()
    
    logger.setLevel(config.log_level.upper())

    if config.log_file:
        try:
            os.makedirs(os.path.dirname(config.log_file), exist_ok=True)

            file_handler = logging.FileHandler(config.log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(config.log_level.upper())

            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {config.log_file}")

            return
        except Exception as e:
            logger.error(f"Failed to setup file logging: {str(e)}")
    
    # Console handler (when no log file specified)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(config.log_level.upper())
    logger.addHandler(console_handler)

    logger.info("Logging initialized with level: %s", config.log_level)

def setup_api_key() -> None:
    """Read the OpenAI API key from a file and set it as an environment variable."""
    try:
        with open(config.api_key_file, "r") as f:
            api_key = f.read().strip()
            os.environ["OPENAI_API_KEY"] = api_key
            logger.info("API key loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load API key: {str(e)}")
        raise
    

def initialize_files() -> None:
    """Ensure all output files and directories exist"""
    # Create temp directory for snippets
    os.makedirs(config.temp_dir, exist_ok=True)
    
    # Initialize answer file if it doesn't exist
    if not os.path.exists(config.answers_file):
        answer_columns = [
            "Chart", "Alert_ID", "Tool", "Resource", "Query", "LLM", 
            "Input_Tokens", "Refactored_YAML", "Output_Tokens", "Original_YAML"
        ]

        pd.DataFrame(columns=answer_columns).to_csv(config.answers_file, index=False)
        logger.info(f"Created new answer file: {config.answers_file}")
    
    # Initialize failed queries file
    if not os.path.exists(config.failed_queries_file):
        failed_columns = [
            "Chart", "Alert_ID", "Tool", "LLM", "Input_Tokens",
            "Original_YAML", "Refactored_YAML", "Error_Type"
        ]

        pd.DataFrame(columns=failed_columns).to_csv(config.failed_queries_file, index=False)
        logger.info(f"Created new failed queries file: {config.failed_queries_file}")
    
    # Initialize evaluation file
    if not os.path.exists(config.evaluation_file):
        eval_columns = [
            "Chart", "Alert_ID", "Tool", "LLM", "Syntax_Improvement",
            "Changed_Lines", "Added_Lines", "Removed_Lines"
        ]

        pd.DataFrame(columns=eval_columns).to_csv(config.evaluation_file, index=False)
        logger.info(f"Created new evaluation file: {config.evaluation_file}")

def clean_llm_response(response: str) -> str:
    """Clean up the LLM response by removing common formatting markers."""
    markers = ["---", "```yaml", "```"]
    for marker in markers:
        response = response.replace(marker, "")

    return response.strip()

def parse_yaml_response(response: str) -> Any:
    """Attempt to parse the YAML response"""
    try:
        return yaml.safe_load(response)
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse YAML response: {str(e)}")
        return "Failed to parse YAML."
    except Exception as e:
        logger.error(f"Unexpected error parsing YAML: {str(e)}")
        return "Failed to parse YAML."

def process_chatgpt_response(client: OpenAI, row: Dict) -> Dict:
    """Process a single row with ChatGPT and return the results."""
    chart_name = row["Chart"]
    alert_id = row["Alert_ID"]
    
    logger.debug(f"Processing {chart_name} - {alert_id}")
    
    try:
        completion = client.chat.completions.create(
            model=config.model_name,
            messages=[
                {"role": "system", "content": row["Query"]},
                {"role": "user", "content": row["Original_YAML"]}
            ]
        )
        
        answer = clean_llm_response(completion.choices[0].message.content)
        answer_dict = parse_yaml_response(answer)
        
        return {
            "Chart": chart_name,
            "Alert_ID": alert_id,
            "Tool": row["Tool"],
            "Resource": row["Resource"],
            "Query": row["Query"],
            "LLM": f"ChatGPT-{config.model_name}",
            "Input_Tokens": completion.usage.prompt_tokens,
            "Refactored_YAML": answer_dict,
            "Output_Tokens": completion.usage.completion_tokens,
            "Original_YAML": row["Original_YAML"]
        }
        
    except Exception as e:
        logger.error(f"Error processing {chart_name} - {alert_id}: {str(e)}")
        return {
            "Chart": chart_name,
            "Alert_ID": alert_id,
            "Tool": row["Tool"],
            "Resource": row["Resource"],
            "Query": row["Query"],
            "LLM": f"ChatGPT-{config.model_name}",
            "Input_Tokens": 0,
            "Refactored_YAML": "Failed to generate a response.",
            "Output_Tokens": 0,
            "Original_YAML": row["Original_YAML"]
        }

def save_results_batch(results: List[Dict]) -> None:
    """Save a batch of results to the answer file."""
    try:
        existing_df = pd.read_csv(config.answers_file)
        new_df = pd.DataFrame(results)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(config.answers_file, index=False)
        logger.debug(f"Saved {len(results)} results to {config.answers_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        raise

def process_single_query(client: OpenAI, df: pd.DataFrame, idx: int) -> Dict:
    """Wrapper function for processing a single query, used for threading."""
    row = df.iloc[idx].to_dict()
    return process_chatgpt_response(client, row)

def query_chatgpt(start_idx: int, end_idx: int) -> None:
    """Query ChatGPT to generate fixes for the specified range of queries."""
    logger.info(f"Processing queries from index {start_idx} to {end_idx} with {config.max_workers} workers")
    
    try:
        df = pd.read_csv(config.queries_file)
        client = OpenAI()
        results = []
        
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # Create a future for each query
            futures = {
                executor.submit(process_single_query, client, df, idx): idx
                for idx in range(start_idx, end_idx)
            }
            
            # Process completed futures
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing queries"):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Save results in batches
                    if len(results) >= config.batch_size:
                        save_results_batch(results)
                        results = []
                except Exception as e:
                    logger.error(f"Error processing future: {str(e)}")
                    continue
                
        # Save any remaining results
        if results:
            save_results_batch(results)
            
    except Exception as e:
        logger.error(f"Error in query_chatgpt: {str(e)}")
        raise


def main():
    """Main function"""
    global config
    
    args = parse_args()
    config = Config(
        log_file = args.log_file,
        log_level = args.log_level,
        model_name = args.model,
        max_workers = args.max_workers
    )
    
    setup_logging()
    setup_api_key()
    initialize_files()
    
    try:
        check_bounds(config.queries_file, args.start_index, args.end_index)
        query_chatgpt(args.start_index, args.end_index or len(pd.read_csv(config.queries_file)))
        logger.info("Query processing completed successfully")       
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()