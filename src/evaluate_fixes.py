from parse_tools import get_ckv_resource_path, get_datree_path, CheckovLookup, DatreeLookup
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import logging
import argparse
from pathlib import Path
from multiprocessing import cpu_count
from dataclasses import dataclass
from helpers import check_bounds, CheckovLookup, DatreeLookup
import yaml
import os
import json
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)
config = None

# Specify directories for reference
script_dir = Path(__file__).parent.resolve()
root_dir = script_dir.parent
results_dir = os.path.join(root_dir, "experiment2", "results")


@dataclass
class Config:
    """Configuration class for the evaluation script."""
    # IO files
    answers_file: str = os.path.join(results_dir, "llm_chatgpt_answers.csv")
    output_file: str = os.path.join(results_dir, "chatgpt_fix.csv")

    # Directories
    snippets_dir: str = os.path.join(root_dir, "experiment2", "snippets")
    manual_analysis_dir: str = os.path.join(results_dir, "manual_analysis")
    checkov_output_dir: str = os.path.join(manual_analysis_dir, "tool_output", "checkov")
    datree_output_dir: str = os.path.join(manual_analysis_dir, "tool_output", "datree")
    plots_dir: str = os.path.join(manual_analysis_dir, "plots")
    
    # Logging
    log_file: Optional[str] = None
    log_level: str = "INFO"
    
    # Processing
    batch_size: int = 20
    max_workers: int = max(cpu_count(), 1)
    llm_name: str = "chatgpt"  # Default LLM to evaluate


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate LLM-generated fixes for Helm chart vulnerabilities."
    )
    
    # Processing arguments
    parser.add_argument("--start-index", type = int, default = 0, help = "Starting index for processing queries")
    parser.add_argument("--end-index", type = int, help = "Ending index for processing queries")
    parser.add_argument("--llm", type = str, default = "chatgpt", help = "LLM to evaluate (default: chatgpt)")
    
    # Logging arguments
    parser.add_argument("--log-file", help = "Path to log file", default = None)
    parser.add_argument("--log-level", help = "Logging level", default = "INFO", choices = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    
    # Threading arguments
    parser.add_argument("--max-workers", type = int, help = "Maximum number of worker threads", default = max(cpu_count(), 1))
    
    return parser.parse_args()


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


def initialize_files() -> None:
    """Ensure all output files and directories exist."""

    os.makedirs(config.snippets_dir, exist_ok = True)
    os.makedirs(config.manual_analysis_dir, exist_ok = True)
    os.makedirs(config.plots_dir, exist_ok = True)

    # Create subdirectories for each tool
    os.makedirs(config.checkov_output_dir, exist_ok = True)
    os.makedirs(config.datree_output_dir, exist_ok = True)

    
    if not os.path.exists(config.output_file):
        pd.DataFrame(columns=[
            "Chart", "Alert_ID", "Tool", "Resource", "LLM", "Fixed",
            "Added_Lines", "Changed_Lines", "Removed_Lines"
        ]).to_csv(config.output_file, index=False)
        logger.info(f"Created new output file: {config.output_file}")


def add_std_id() -> None:
    """Add standard IDs to the LLM answers file by mapping alert IDs.
    
    Reads the answers file, maps each alert to its standard ID using the appropriate
    lookup table, and saves the results back to the file.
    """
    try:
        df = pd.read_csv(config.answers_file)
        std_checks = []
        
        for _, row in df.iterrows():
            if row["Tool"] == "checkov":
                std_checks.append(CheckovLookup.get_value(row["Alert_ID"]))
            elif row["Tool"] == "datree":
                std_checks.append(DatreeLookup.get_value(row["Alert_ID"]))
            else:
                std_checks.append("")
        
        df["Standard_ID"] = std_checks
        df.to_csv(config.answers_file, index=False)
        logger.info("Successfully added standard IDs to answers file")
    except Exception as e:
        logger.error(f"Failed to add standard IDs: {str(e)}")
        raise


def check_paths(paths: Dict[str, str], resource: str) -> bool:
    """Check if the paths match the resource."""
    try:
        aux_resource = [paths["resource_path"], paths["obj_path"]]
        resource = ast.literal_eval(resource)
        return aux_resource == resource
    except (ValueError, SyntaxError, KeyError) as e:
        logger.error(f"Error comparing paths: {str(e)}")
        return False


def parse_checkov_results(result_path: str, chart_name: str, idx: int, alert_id: str, resource: str) -> str:
    """Parse Checkov results to determine if vulnerability was fixed."""
    try:
        with open(result_path, 'r', encoding="utf-8") as file:
            results = json.load(file)
        
        temp_dir = os.path.join(config.snippets_dir, config.llm_name)
        refactored_path = os.path.join(temp_dir, f"refactored_{idx}.yaml")
        
        with open(refactored_path, "r", encoding="utf-8") as file:
            template = list(yaml.safe_load_all(file))
        
        if "results" in results and "failed_checks" in results["results"]:
            for check in results["results"]["failed_checks"]:
                if check['check_id'] == alert_id:
                    paths = get_ckv_resource_path(check, template)
                    if check_paths(paths, resource):
                        return "Not_Fixed"
        return "Fixed"
    except Exception as e:
        logger.error(f"Error parsing Checkov results for {chart_name} (index {idx}): {str(e)}")
        return "Not_Fixed"


def parse_datree_results(result_path: str, chart_name: str, idx: int, alert_id: str, resource: str) -> str:
    """Parse Datree results to determine if vulnerability was fixed."""
    try:
        with open(result_path, 'r', encoding="utf-8") as file:
            results = json.load(file)
        
        if "policyValidationResults" in results and results["policyValidationResults"]:
            for check in results["policyValidationResults"][0]["ruleResults"]:
                if check['identifier'] == alert_id:
                    for occurrence in check["occurrencesDetails"]:
                        paths = get_datree_path(occurrence)
                        if check_paths(paths, resource):
                            return "Not_Fixed"
        return "Fixed"
    except Exception as e:
        logger.error(f"Error parsing Datree results for {chart_name} (index {idx}): {str(e)}")
        return "Not_Fixed"


def run_static_analysis(tool: str, refactored_path: str, chart_name: str, alert_id: int) -> str:
    """Run static analysis tool on the refactored YAML."""

    if tool == "checkov":
        result_path = os.path.join(config.checkov_output_dir, f"{chart_name}_{alert_id}.json")
    elif tool == "datree":
        result_path = os.path.join(config.datree_output_dir, f"{chart_name}_{alert_id}.json")

    
    try:
        if tool == "checkov":
            cmd = f"checkov -f {refactored_path} --quiet --compact --framework kubernetes -o json > {result_path}"
        elif tool == "datree":
            cmd = f"helm datree test {refactored_path} --only-k8s-files --quiet --output json > {result_path}"
        
        exit_code = os.system(cmd)
        if exit_code != 0:
            logger.warning(f"Static analysis tool {tool} returned non-zero exit code for {chart_name}_{alert_id}")
        
        return result_path
    except Exception as e:
        logger.error(f"Error running {tool} for {chart_name}_{alert_id}: {str(e)}")


def calculate_yaml_diff(original_path: str, refactored_path: str) -> Tuple[int, int, int]:
    """Calculate differences between original and refactored YAML files."""
    try:
        # Count lines
        with open(original_path, "r", encoding="utf-8") as f:
            original_lines = f.readlines()
        with open(refactored_path, "r", encoding="utf-8") as f:
            refactored_lines = f.readlines()
        
        len_original = len(original_lines)
        len_refactored = len(refactored_lines)
        
        added = max(0, len_refactored - len_original)
        removed = max(0, len_original - len_refactored)
        
        # Load YAML content for detailed comparison
        with open(original_path, "r", encoding="utf-8") as f:
            original_yaml = yaml.safe_load(f)
        with open(refactored_path, "r", encoding="utf-8") as f:
            refactored_yaml = yaml.safe_load(f)
        
        changed = 0
        if not isinstance(original_yaml, dict) or not isinstance(refactored_yaml, dict):
            return added, changed, removed

        for key in original_yaml:
            if key in refactored_yaml:
                if isinstance(original_yaml[key], dict):
                    for sub_key in original_yaml[key]:
                        if sub_key in refactored_yaml[key]:
                            if original_yaml[key][sub_key] != refactored_yaml[key][sub_key]:
                                changed += 1
                elif original_yaml[key] != refactored_yaml[key]:
                    changed += 1
        
        return added, changed, removed
    except Exception as e:
        logger.error(f"Error calculating YAML differences: {str(e)}")
        return 0, 0, 0


def process_single_fix(idx: int, row: Dict[str, Any]) -> Dict[str, Any]:
    """Process and evaluate a single fix generated by the LLM."""

    chart_name = row["Chart"]
    alert_id = row["Alert_ID"]
    tool = row["Tool"]
    resource = row["Resource"]
    
    logger.debug(f"Processing {chart_name} - {alert_id} (index {idx})")
    
    # Prepare file paths
    refactored_path = os.path.join(config.snippets_dir, config.llm_name, f"{chart_name}_{alert_id}_refactored.yaml")
    original_path = os.path.join(config.snippets_dir, config.llm_name, f"{chart_name}_{alert_id}_original.yaml")

    
    # Initialize result dictionary
    result = {
        "Chart": chart_name,
        "Alert_ID": alert_id,
        "Tool": tool,
        "Resource": resource,
        "LLM": f"{config.llm_name}",
        "Fixed": "Not_Fixed",
        "Added_Lines": 0,
        "Changed_Lines": 0,
        "Removed_Lines": 0,
        "Standard_ID": row.get("Standard_ID", "")
    }
    
    # Check if the fix failed to generate
    if str(row["Refactored_YAML"]).startswith("Failed"):
        return result
    
    # Prepare temp file paths
    temp_dir = os.path.join(config.snippets_dir, config.llm_name)
    os.makedirs(temp_dir, exist_ok=True)
    refactored_path = os.path.join(temp_dir, f"refactored_{idx}.yaml")
    original_path = os.path.join(temp_dir, f"original_{idx}.yaml")
    
    try:
        # Write YAML files
        refactored_yaml = yaml.safe_load(row["Refactored_YAML"])
        original_yaml = yaml.safe_load(row["Original_YAML"])
        
        with open(refactored_path, "w", encoding="utf-8") as f:
            yaml.dump(refactored_yaml, f)
        with open(original_path, "w", encoding="utf-8") as f:
            yaml.dump(original_yaml, f, default_flow_style=False)
        
        # Run static analysis
        result_path = run_static_analysis(tool, refactored_path, chart_name, alert_id)
        
        # Check if fix was successful
        if tool == "checkov":
            fixed = parse_checkov_results(result_path, chart_name, idx, alert_id, resource)
        elif tool == "datree":
            fixed = parse_datree_results(result_path, chart_name, idx, alert_id, resource)
        else:
            fixed = "Not_Fixed"
        
        # Calculate YAML differences
        added, changed, removed = calculate_yaml_diff(original_path, refactored_path)
        
        # Update result
        result.update({
            "Fixed": fixed,
            "Added_Lines": added,
            "Changed_Lines": changed,
            "Removed_Lines": removed
        })
        
    except Exception as e:
        logger.error(f"Error processing fix {idx}: {str(e)}")
    
    finally:
        # Clean up temporary files
        try:
            os.remove(refactored_path)
            os.remove(original_path)
        except OSError:
            pass
    
    return result


def save_results_batch(results: List[Dict[str, Any]]) -> None:
    """Save a batch of results to the output file.
    
    Args:
        results: List of result dictionaries to save
    """
    try:
        existing_df = pd.read_csv(config.output_file)
        new_df = pd.DataFrame(results)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(config.output_file, index=False)
        logger.debug(f"Saved {len(results)} results to {config.output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        raise


def evaluate_fixes(start_idx: int, end_idx: Optional[int]) -> None:
    """Evaluate LLM-generated fixes for a range of queries."""

    try:
        add_std_id()
    except Exception as e:
        logger.error(f"Failed to add standard IDs, proceeding anyway: {str(e)}")

    logger.info(f"Evaluating fixes from index {start_idx} to {end_idx} with {config.max_workers} workers")
    
    try:
        df = pd.read_csv(config.answers_file)
        if end_idx is None or end_idx > len(df):
            end_idx = len(df)
        
        results = []
        
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(process_single_fix, idx, df.iloc[idx].to_dict()): idx
                for idx in range(start_idx, end_idx)
            }
            
            for future in tqdm(as_completed(futures), total = len(futures), desc = "Evaluating fixes"):
                try:
                    result = future.result()
                    results.append(result)
                    
                    if len(results) >= config.batch_size:
                        save_results_batch(results)
                        results = []
                except Exception as e:
                    logger.error(f"Error processing future: {str(e)}")
                    continue
        
        if results:
            save_results_batch(results)
            
    except Exception as e:
        logger.error(f"Error in evaluate_fixes: {str(e)}")
        raise


def generate_analysis_report() -> None:
    """Generate statistics and visualizations."""
    try:
        df = pd.read_csv(config.output_file)
        
        if df.empty:
            logger.warning("No data available for analysis.")
            return
        
        # Ensure necessary columns exist
        required_columns = {'Fixed', 'Tool', 'Added_Lines', 'Changed_Lines', 'Removed_Lines'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            logger.warning(f"Missing columns in dataset: {missing_columns}")
            return
        
        # Basic statistics
        total_fixes = len(df)
        fixed_count = (df['Fixed'] == 'Fixed').sum()
        fix_rate = (fixed_count / total_fixes * 100) if total_fixes > 0 else 0

        stats = {
            "Total_Fixes_Attempted": total_fixes,
            "Successfully_Fixed": fixed_count,
            "Fix_Rate_Percentage": round(fix_rate, 2),
        }
        # Convert numpy types to native Python types
        stats = {key: float(value) if isinstance(value, (np.float32, np.float64)) else int(value) if isinstance(value, (np.int32, np.int64)) else value for key, value in stats.items()}


        # Compute statistics on code changes
        change_columns = ['Added_Lines', 'Changed_Lines', 'Removed_Lines']
        change_stats = df[change_columns].agg(['mean', 'median', 'std']).to_dict()
        for col in change_columns:
            stats[f"{col}_Mean"] = round(change_stats[col]['mean'], 2)
            stats[f"{col}_Median"] = round(change_stats[col]['median'], 2)
            stats[f"{col}_StdDev"] = round(change_stats[col]['std'], 2)
        
        # Save numerical stats
        stats_path = os.path.join(config.manual_analysis_dir, "fix_statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        # Tool-specific success rate
        tool_stats = df.groupby('Tool')['Fixed'].value_counts(normalize=True).unstack(fill_value=0) * 100
        tool_stats.to_csv(os.path.join(config.manual_analysis_dir, "tool_stats.csv"))
        
        # Standard ID analysis (top unfixed issues)
        if 'Standard_ID' in df.columns:
            top_issues = df[df['Fixed'] == 'Not_Fixed']['Standard_ID'].value_counts().head(10)
            if not top_issues.empty:
                top_issues.to_csv(os.path.join(config.manual_analysis_dir, "top_unfixed_issues.csv"))
                
                # Plot top unfixed issues
                plt.figure(figsize=(12, 6))
                sns.barplot(x=top_issues.index, y=top_issues.values, palette="coolwarm")
                plt.title('Top 10 Unfixed Issues')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(config.plots_dir, 'top_unfixed_issues.png'))
                plt.close()
        
        # Plot fix rates by tool
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='Tool', hue='Fixed', palette="pastel")
        plt.title('Fix Success by Tool')
        plt.tight_layout()
        plt.savefig(os.path.join(config.plots_dir, 'fix_success_by_tool.png'))
        plt.close()
        
        # Plot distribution of changes
        plt.figure(figsize=(12, 6))
        df[change_columns].plot.kde()
        plt.title('Distribution of Code Changes')
        plt.tight_layout()
        plt.savefig(os.path.join(config.plots_dir, 'change_distribution.png'))
        plt.close()
        
        logger.info(f"Generated analysis report in {config.manual_analysis_dir}")
    except Exception as e:
        logger.error(f"Failed to generate analysis report: {str(e)}")


def main() -> None:
    """Main function to run the evaluation pipeline."""
    global config
    
    args = parse_args()
    config = Config(
        log_file=args.log_file,
        log_level=args.log_level,
        max_workers=args.max_workers,
        llm_name=args.llm
    )
    
    setup_logging()
    initialize_files()
    
    try:
        check_bounds(config.answers_file, args.start_index, args.end_index)
        evaluate_fixes(args.start_index, args.end_index)
        logger.info("Evaluation completed successfully")

        # Generate analysis report
        generate_analysis_report()

        # Remove temporary files
        os.system(f"rm -rf {config.snippets_dir}")
        
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
