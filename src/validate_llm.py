import os
import json
import yaml
import logging
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from difflib import Differ
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional


logger = logging.getLogger(__name__)
config = None 

# Specify directories for reference
script_dir = Path(__file__).parent.resolve()
root_dir = script_dir.parent


@dataclass
class Config:
    # IO files and directories
    queries_file = os.path.join(root_dir, "experiment2", "results", "llm_queries.csv")
    answers_file = os.path.join(root_dir, "experiment2", "results", "llm_chatgpt_answers.csv")
    failed_queries_file = os.path.join(root_dir, "experiment2", "results", "failed_queries.csv")
    evaluation_file = os.path.join(root_dir, "experiment2", "results", "llm_evaluation.csv")
    temp_dir = os.path.join(root_dir, "temp_snippets")
    
    # Logging
    log_file: Optional[str] = None
    log_level: Optional[str] = "INFO"
    
    # Evaluation
    kubeconform_path = os.path.join(root_dir, "bin", "kubeconform")

def parse_args():
    parser = argparse.ArgumentParser (description="Query ChatGPT for Kubernetes YAML refactoring suggestions and evaluate results.")
    
    # Logging 
    parser.add_argument("--log-file", help="Path to log file", default = None)
    parser.add_argument("--log-level", help="Logging level", default = "INFO", choices = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    
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


def save_yaml_snippet(yaml_content: Any, prefix: str) -> None:
    """Save YAML snippet to a temporary file."""
    try:
        filepath = os.path.join(config.temp_dir, f"{prefix}_snippet.yaml")
        with open(filepath, "w") as f:
            if isinstance(yaml_content, dict):
                yaml.dump(yaml_content, f)
            else:
                f.write(str(yaml_content))
        logger.debug(f"Saved {prefix} snippet to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save {prefix} snippet: {str(e)}")
        raise

def run_kubeconform(filepath: str) -> Dict:
    """Run kubeconform on a YAML file and return the results."""
    try:
        result_file = os.path.join(config.temp_dir, "syntax.json")
        cmd = [
            f'{config.kubeconform_path}',
            "-summary",
            "-output", "json",
            filepath
        ]
        
        with open(result_file, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, check=True)
        
        with open(result_file, "r") as f:
            return json.load(f)
    except subprocess.CalledProcessError as e:
        logger.error(f"kubeconform failed: {e.stderr.decode()}")
        return {"summary": {"invalid": 1}}
    except Exception as e:
        logger.error(f"Error running kubeconform: {str(e)}")
        return {"summary": {"invalid": 1}}

def validate_syntax(original: Any, refactored: Any) -> int:
    """Validate the syntax of YAML snippets and return improvement score."""
    try:
        # Save snippets to temp files
        save_yaml_snippet(original, "original")
        save_yaml_snippet(refactored, "refactored")
        
        # Validate original snippet
        original_result = run_kubeconform(
            os.path.join(config.temp_dir, "original_snippet.yaml")
        )
        
        # Validate refactored snippet
        refactored_result = run_kubeconform(
            os.path.join(config.temp_dir, "refactored_snippet.yaml")
        )
        
        # Calculate improvement
        improvement = original_result["summary"]["invalid"] - refactored_result["summary"]["invalid"]
        return max(improvement, 0)
        
    except Exception as e:
        logger.error(f"Syntax validation failed: {str(e)}")
        return 0

def compare_snippets(original: Any, refactored: Any) -> Tuple[int, int, int]:
    """Compare original and refactored snippets, returning change metrics."""
    try:
        # Save snippets to temp files
        save_yaml_snippet(original, "original")
        save_yaml_snippet(refactored, "refactored")
        
        # Read files for comparison
        with open(os.path.join(config.temp_dir, "original_snippet.yaml"), "r") as f:
            original_lines = f.readlines()
        
        with open(os.path.join(config.temp_dir, "refactored_snippet.yaml"), "r") as f:
            refactored_lines = f.readlines()
        
        # Line differences
        added = max(len(refactored_lines) - len(original_lines), 0)
        removed = max(len(original_lines) - len(refactored_lines), 0)
        changed = 0
        
        # Difflib comparisson
        differ = Differ()
        diff = list(differ.compare(original_lines, refactored_lines))
        
        for i, line in enumerate(diff):
            if line.startswith('- '):
                if i+1 < len(diff) and diff[i+1].startswith('+ '):
                    changed += 1
        
        return changed, added, removed
        
    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}")
        return 0, 0, 0

def evaluate_responses() -> None:
    """Evaluate all responses in the answers file."""
    try:
        df = pd.read_csv(config.answers_file)
        evaluation_results = []
        
        for _, row in tqdm(df.iterrows(), total = len(df), desc = "Evaluating responses"):
            if row["Refactored_YAML"] in ["Failed to parse YAML.", "Failed to generate a response."]:
                continue
                
            try:
                # Get YAML content
                original = yaml.safe_load(row["Original_YAML"])
                refactored = yaml.safe_load(row["Refactored_YAML"])
                
                # Perform evaluations
                syntax_improvement = validate_syntax(original, refactored)
                changed, added, removed = compare_snippets(original, refactored)
                
                evaluation_results.append({
                    "Chart": row["Chart"],
                    "Alert_ID": row["Alert_ID"],
                    "Tool": row["Tool"],
                    "LLM": row["LLM"],
                    "Syntax_Improvement": syntax_improvement,
                    "Changed_Lines": changed,
                    "Added_Lines": added,
                    "Removed_Lines": removed
                })
                
            except Exception as e:
                logger.error(f"Failed to evaluate {row['Chart']} - {row['Alert_ID']}: {str(e)}")
                continue
                
        # Save evaluation results
        if evaluation_results:
            eval_df = pd.DataFrame(evaluation_results)
            existing_eval_df = pd.read_csv(config.evaluation_file)
            combined_df = pd.concat([existing_eval_df, eval_df], ignore_index=True)
            combined_df.to_csv(config.evaluation_file, index=False)
            logger.info(f"Saved {len(evaluation_results)} evaluation results")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


def main():
    global config

    args = parse_args()
    config = Config(
        log_file = args.log_file,
        log_level = args.log_level
    )

    setup_logging()
    
    evaluate_responses()
    
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()