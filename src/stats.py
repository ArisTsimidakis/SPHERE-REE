import os
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
config = None 

# Specify directories for reference
script_dir = Path(__file__).parent.resolve()
root_dir = script_dir.parent

@dataclass
class Config:
    # Input files
    template_stats_file: str = os.path.join(root_dir, "experiment", "results", "template_stats.csv")
    failed_checks_file: str = os.path.join(root_dir, "experiment", "results", "failed_checks.csv")
    queries_file: str = os.path.join(root_dir, "experiment", "results", "llm_queries.csv")
    answers_file: str = os.path.join(root_dir, "experiment", "results", "llm_chatgpt_answers.csv")
    failed_queries_file: str = os.path.join(root_dir, "experiment", "results", "failed_queries.csv")
    evaluation_file: str = os.path.join(root_dir, "experiment", "results", "llm_evaluation.csv")
    
    # Output options
    output_dir: str = os.path.join(root_dir, "experiment", "stats")
    generate_plots: bool = True
    
    # Logging
    log_file: Optional[str] = None
    log_level: Optional[str] = "INFO"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate comprehensive statistics for the Kubernetes YAML refactoring pipeline.")
    
    # Input/output options
    parser.add_argument("--output-dir", help="Directory to save statistics and plots", default=None)
    parser.add_argument("--no-plots", help="Disable plot generation", action="store_true")
    
    # Logging 
    parser.add_argument("--log-file", help="Path to log file", default=None)
    parser.add_argument("--log-level", help="Logging level", default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    
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

def ensure_output_dir() -> None:
    """Ensure the output directory exists"""
    try:
        os.makedirs(config.output_dir, exist_ok=True)
        logger.info(f"Output directory ready: {config.output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory: {str(e)}")
        raise

def save_table(data: Dict[str, any], filename: str, format: str = "grid") -> None:
    """Save a table to a text file"""
    try:
        path = os.path.join(config.output_dir, filename)
        with open(path, "w") as f:
            f.write(tabulate(data, headers="keys", tablefmt=format))
        logger.info(f"Saved table to {path}")
    except Exception as e:
        logger.error(f"Failed to save table {filename}: {str(e)}")

def save_plot(fig, filename: str) -> None:
    """Save a matplotlib figure"""
    if not config.generate_plots:
        return
        
    try:
        path = os.path.join(config.output_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved plot to {path}")
    except Exception as e:
        logger.error(f"Failed to save plot {filename}: {str(e)}")

def describe_numerical(df: pd.DataFrame, column: str) -> Dict[str, float]:
    """Generate descriptive statistics for a numerical column"""
    stats = df[column].describe().to_dict()
    stats["median"] = df[column].median()
    stats["missing"] = df[column].isna().sum()
    return stats

def analyze_template_stats() -> None:
    """Analyze statistics from the template generation phase"""
    try:
        logger.info("Analyzing template statistics...")
        df = pd.read_csv(config.template_stats_file)
        
        # Basic stats
        total_charts = len(df)
        status_counts = df["status"].value_counts().to_dict()
        
        # Numerical stats
        lines_stats = describe_numerical(df[df["lines"] > 0], "lines")
        containers_stats = describe_numerical(df[df["containers"] > 0], "containers")
        chars_stats = describe_numerical(df[df["characters"] > 0], "characters")
        
        # Save results
        results = {
            "Total Charts": [total_charts],
            "Correct Templates": [status_counts.get("Correct", 0)],
            "Empty Templates": [status_counts.get("Empty", 0)],
            "Error Templates": [status_counts.get("Error", 0)],
            "Avg Lines (non-empty)": [lines_stats["mean"]],
            "Avg Containers (non-zero)": [containers_stats["mean"]],
            "Avg Characters (non-empty)": [chars_stats["mean"]]
        }
        
        save_table(results, "template_stats_summary.txt")
        
        # Generate plots
        if config.generate_plots:
            # Status distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            df["status"].value_counts().plot(kind="bar", ax=ax)
            ax.set_title("Template Status Distribution")
            save_plot(fig, "template_status_distribution.png")
            
            # Lines distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(df[df["lines"] > 0]["lines"], bins=30, kde=True, ax=ax)
            ax.set_title("Lines of YAML Distribution (non-empty templates)")
            save_plot(fig, "template_lines_distribution.png")
            
        logger.info("Template statistics analysis completed")
        return results
        
    except Exception as e:
        logger.error(f"Failed to analyze template stats: {str(e)}")
        raise

def analyze_failed_checks() -> None:
    """Analyze statistics from the failed checks phase"""
    try:
        logger.info("Analyzing failed checks...")
        df = pd.read_csv(config.failed_checks_file)
        
        # Basic stats
        total_failed_checks = len(df)
        unique_charts = df["Chart"].nunique()
        
        # Tool distribution
        tool_dist = df["Tool"].value_counts().to_dict()
        
        # Alert ID distribution
        top_alerts = df["Alert_ID"].value_counts().head(20).to_dict()
        
        # Resource type distribution
        resource_types = df["Resource"].apply(lambda x: x.split("/")[0] if isinstance(x, str) else "Unknown")
        resource_dist = resource_types.value_counts().to_dict()
        
        # YAML size stats
        yaml_size_stats = describe_numerical(df, "Characters")
        
        # Save results
        results = {
            "Metric": ["Total Failed Checks", "Unique Charts Affected", 
                      "Avg YAML Size (chars)", "Median YAML Size (chars)",
                      "Most Common Tool", "Most Common Alert"],
            "Value": [total_failed_checks, unique_charts,
                     yaml_size_stats["mean"], yaml_size_stats["median"],
                     max(tool_dist, key=tool_dist.get), max(top_alerts, key=top_alerts.get)]
        }
        
        save_table(results, "failed_checks_summary.txt")
        
        # Detailed tables
        save_table(pd.DataFrame.from_dict(tool_dist, orient="index", columns=["Count"]), 
                  "failed_checks_tool_distribution.txt")
        save_table(pd.DataFrame.from_dict(top_alerts, orient="index", columns=["Count"]), 
                  "failed_checks_top_alerts.txt")
        save_table(pd.DataFrame.from_dict(resource_dist, orient="index", columns=["Count"]), 
                  "failed_checks_resource_types.txt")
        
        # Generate plots
        if config.generate_plots:
            # Tool distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            df["Tool"].value_counts().plot(kind="bar", ax=ax)
            ax.set_title("Failed Checks by Tool")
            save_plot(fig, "failed_checks_tool_distribution.png")
            
            # Top alerts
            fig, ax = plt.subplots(figsize=(12, 6))
            df["Alert_ID"].value_counts().head(20).plot(kind="bar", ax=ax)
            ax.set_title("Top 20 Alert IDs")
            save_plot(fig, "failed_checks_top_alerts.png")
            
            # YAML size distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(df["Characters"], bins=30, kde=True, ax=ax)
            ax.set_title("Failed Check YAML Size Distribution")
            save_plot(fig, "failed_checks_yaml_size.png")
            
        logger.info("Failed checks analysis completed")
        return results
        
    except Exception as e:
        logger.error(f"Failed to analyze failed checks: {str(e)}")
        raise

def analyze_llm_queries() -> None:
    """Analyze statistics from the LLM query generation phase"""
    try:
        logger.info("Analyzing LLM queries...")
        df = pd.read_csv(config.queries_file)
        
        # Basic stats
        total_queries = len(df)
        unique_charts = df["Chart"].nunique()
        
        # Query length stats
        df["Query_Length"] = df["Query"].apply(len)
        query_len_stats = describe_numerical(df, "Query_Length")
        
        # YAML size stats
        df["YAML_Length"] = df["Original_YAML"].apply(lambda x: len(str(x)))
        yaml_len_stats = describe_numerical(df, "YAML_Length")
        
        # Alert ID distribution
        alert_dist = df["Alert_ID"].value_counts().head(20).to_dict()
        
        # Save results
        results = {
            "Total Queries": [total_queries],
            "Unique Charts": [unique_charts],
            "Avg Query Length (chars)": [query_len_stats["mean"]],
            "Median Query Length": [query_len_stats["median"]],
            "Avg YAML Length (chars)": [yaml_len_stats["mean"]],
            "Median YAML Length": [yaml_len_stats["median"]]
        }
        
        save_table(results, "llm_queries_summary.txt")
        save_table(pd.DataFrame.from_dict(alert_dist, orient="index", columns=["Count"]), 
                  "llm_queries_alert_distribution.txt")
        
        # Generate plots
        if config.generate_plots:
            # Query length distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(df["Query_Length"], bins=30, kde=True, ax=ax)
            ax.set_title("LLM Query Length Distribution")
            save_plot(fig, "llm_queries_length_distribution.png")
            
            # YAML length distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(df["YAML_Length"], bins=30, kde=True, ax=ax)
            ax.set_title("Original YAML Length Distribution")
            save_plot(fig, "llm_queries_yaml_length.png")
            
        logger.info("LLM queries analysis completed")
        return results
        
    except Exception as e:
        logger.error(f"Failed to analyze LLM queries: {str(e)}")
        raise

def analyze_llm_responses() -> None:
    """Analyze statistics from the LLM response phase"""
    try:
        logger.info("Analyzing LLM responses...")
        answers_df = pd.read_csv(config.answers_file)
        failed_df = pd.read_csv(config.failed_queries_file)
        
        # Basic stats
        total_queries = len(answers_df)
        
        # Check if 'Refactored_YAML' column exists and count successful responses
        if 'Refactored_YAML' in answers_df.columns:
            successful = len(answers_df[~answers_df["Refactored_YAML"].isin(
                ["Failed to parse YAML.", "Failed to generate a response."])])
            failed = total_queries - successful
        else:
            logger.warning("'Refactored_YAML' column not found in answers file")
            successful = 0
            failed = total_queries
        
        # Token usage stats (check if columns exist)
        input_token_stats = {}
        output_token_stats = {}
        
        if 'Input_Tokens' in answers_df.columns:
            input_token_stats = describe_numerical(answers_df, "Input_Tokens")
        else:
            logger.warning("'Input_Tokens' column not found in answers file")
            
        if 'Output_Tokens' in answers_df.columns:
            output_token_stats = describe_numerical(answers_df, "Output_Tokens")
        else:
            logger.warning("'Output_Tokens' column not found in answers file")
        
        # Failure analysis
        failure_reasons = {}
        failure_by_chart = {}
        failure_by_alert = {}
        
        if not failed_df.empty and 'Error_Type' in failed_df.columns:
            failure_reasons = failed_df["Error_Type"].value_counts().to_dict()
            failure_by_chart = failed_df["Chart"].value_counts().head(10).to_dict()
            failure_by_alert = failed_df["Alert_ID"].value_counts().head(10).to_dict()
        
        # Save results
        results = {
            "Total Queries": [total_queries],
            "Successful Responses": [f"{successful} ({successful/total_queries:.1%})" if total_queries > 0 else "0"],
            "Failed Responses": [f"{failed} ({failed/total_queries:.1%})" if total_queries > 0 else "0"],
            "Avg Input Tokens": [input_token_stats.get("mean", "N/A")],
            "Median Input Tokens": [input_token_stats.get("median", "N/A")],
            "Avg Output Tokens": [output_token_stats.get("mean", "N/A")],
            "Median Output Tokens": [output_token_stats.get("median", "N/A")],
            "Most Common Failure": [max(failure_reasons, key=failure_reasons.get) if failure_reasons else "N/A"]
        }
        
        save_table(results, "llm_responses_summary.txt")
        
        if failure_reasons:
            save_table(pd.DataFrame.from_dict(failure_reasons, orient="index", columns=["Count"]), 
                      "llm_responses_failure_reasons.txt")
            save_table(pd.DataFrame.from_dict(failure_by_chart, orient="index", columns=["Count"]), 
                      "llm_responses_failure_by_chart.txt")
            save_table(pd.DataFrame.from_dict(failure_by_alert, orient="index", columns=["Count"]), 
                      "llm_responses_failure_by_alert.txt")
        
        # Generate plots
        if config.generate_plots:
            # Success/failure distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            pd.Series({"Successful": successful, "Failed": failed}).plot(kind="bar", ax=ax)
            ax.set_title("LLM Response Success Rate")
            save_plot(fig, "llm_responses_success_rate.png")
            
            # Token usage (only if columns exist)
            if 'Input_Tokens' in answers_df.columns and 'Output_Tokens' in answers_df.columns:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                sns.histplot(answers_df["Input_Tokens"], bins=30, kde=True, ax=ax1)
                ax1.set_title("Input Tokens Distribution")
                sns.histplot(answers_df["Output_Tokens"], bins=30, kde=True, ax=ax2)
                ax2.set_title("Output Tokens Distribution")
                save_plot(fig, "llm_responses_token_usage.png")
            
            if failure_reasons:
                # Failure reasons
                fig, ax = plt.subplots(figsize=(10, 6))
                pd.Series(failure_reasons).plot(kind="bar", ax=ax)
                ax.set_title("LLM Failure Reasons")
                save_plot(fig, "llm_responses_failure_reasons.png")
                
        logger.info("LLM responses analysis completed")
        return results
    except Exception as e:
        logger.error(f"Failed to analyze LLM responses: {str(e)}")
        raise

def analyze_evaluation():
    """Analyze statistics from the evaluation phase"""
    try:
        logger.info("Analyzing evaluation results...")
        df = pd.read_csv(config.evaluation_file)
        
        # Basic stats
        total_evaluations = len(df)
        syntax_improvement = df["Syntax_Improvement"].value_counts().to_dict()
        
        # Change statistics
        changed_lines_stats = describe_numerical(df, "Changed_Lines")
        added_lines_stats = describe_numerical(df, "Added_Lines")
        removed_lines_stats = describe_numerical(df, "Removed_Lines")
        
        # By Alert ID
        changes_by_alert = df.groupby("Alert_ID").agg({
            "Changed_Lines": "mean",
            "Added_Lines": "mean",
            "Removed_Lines": "mean"
        }).sort_values("Changed_Lines", ascending=False).head(10).to_dict()
        
        # Save results
        results = {
            "Total Evaluations": [total_evaluations],
            "Syntax Improved": [syntax_improvement.get(True, 0)],
            "Syntax Worsened": [syntax_improvement.get(False, 0)],
            "Avg Changed Lines": [changed_lines_stats["mean"]],
            "Avg Added Lines": [added_lines_stats["mean"]],
            "Avg Removed Lines": [removed_lines_stats["mean"]],
            "Max Changed Lines": [changed_lines_stats["max"]],
            "Max Added Lines": [added_lines_stats["max"]],
            "Max Removed Lines": [removed_lines_stats["max"]]
        }
        
        save_table(results, "evaluation_summary.txt")
        save_table(pd.DataFrame.from_dict(changes_by_alert), "evaluation_changes_by_alert.txt")
        
        # Generate plots
        if config.generate_plots:
            # Syntax improvement
            fig, ax = plt.subplots(figsize=(8, 6))
            pd.Series(syntax_improvement).plot(kind="bar", ax=ax)
            ax.set_title("Syntax Improvement Results")
            save_plot(fig, "evaluation_syntax_improvement.png")
            
            # Changes distribution
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            sns.histplot(df["Changed_Lines"], bins=30, kde=True, ax=ax1)
            ax1.set_title("Changed Lines Distribution")
            sns.histplot(df["Added_Lines"], bins=30, kde=True, ax=ax2)
            ax2.set_title("Added Lines Distribution")
            sns.histplot(df["Removed_Lines"], bins=30, kde=True, ax=ax3)
            ax3.set_title("Removed Lines Distribution")
            save_plot(fig, "evaluation_changes_distribution.png")
            
        logger.info("Evaluation analysis completed")
        return results
        
    except Exception as e:
        logger.error(f"Failed to analyze evaluation results: {str(e)}")
        raise

def generate_pipeline_summary(results: Dict[str, Dict[str, any]]) -> None:
    """Generate a summary of the entire pipeline"""
    try:
        logger.info("Generating pipeline summary...")
        
        summary_data = []
        
        # Template stats
        template_stats = results.get("template_stats", {})
        summary_data.append({
            "Phase": "Template Generation",
            "Total Charts": template_stats.get("Total Charts", [0])[0],
            "Key Metric": f"{template_stats.get('Correct Templates', [0])[0]} correct templates",
            "Note": f"{template_stats.get('Empty Templates', [0])[0]} empty, {template_stats.get('Error Templates', [0])[0]} errors"
        })
        
        # Failed checks
        failed_checks = results.get("failed_checks", {})
        summary_data.append({
            "Phase": "Failed Checks",
            "Total Charts": failed_checks.get("Unique Charts Affected", [0])[0],
            "Key Metric": f"{failed_checks.get('Total Failed Checks', [0])[0]} checks",
            "Note": f"Avg YAML size: {failed_checks.get('Avg YAML Size (chars)', [0])[0]:.1f} chars"
        })
        
        # LLM Queries
        llm_queries = results.get("llm_queries", {})
        summary_data.append({
            "Phase": "LLM Queries",
            "Total Charts": llm_queries.get("Unique Charts", [0])[0],
            "Key Metric": f"{llm_queries.get('Total Queries', [0])[0]} queries",
            "Note": f"Avg query length: {llm_queries.get('Avg Query Length (chars)', [0])[0]:.1f} chars"
        })
        
        # LLM Responses
        llm_responses = results.get("llm_responses", {})
        summary_data.append({
            "Phase": "LLM Responses",
            "Total Charts": llm_responses.get("Total Queries", [0])[0],
            "Key Metric": llm_responses.get("Successful Responses", ["N/A"])[0],
            "Note": f"Avg tokens: {llm_responses.get('Avg Input Tokens', [0])[0]:.1f} in, {llm_responses.get('Avg Output Tokens', [0])[0]:.1f} out"
        })
        
        # Evaluation
        evaluation = results.get("evaluation", {})
        summary_data.append({
            "Phase": "Evaluation",
            "Total Charts": evaluation.get("Total Evaluations", [0])[0],
            "Key Metric": f"{evaluation.get('Syntax Improved', [0])[0]} improved",
            "Note": f"Avg changes: {evaluation.get('Avg Changed Lines', [0])[0]:.1f} lines"
        })
        
        save_table(summary_data, "pipeline_summary.txt")
        logger.info("Pipeline summary generated")
        
    except Exception as e:
        logger.error(f"Failed to generate pipeline summary: {str(e)}")
        raise

def main():
    global config

    args = parse_args()
    config = Config(
        output_dir=args.output_dir or os.path.join(root_dir, "experiment", "stats"),
        generate_plots=not args.no_plots,
        log_file=args.log_file,
        log_level=args.log_level
    )

    setup_logging()
    ensure_output_dir()
    
    # Run all analyses
    results = {
        "template_stats": analyze_template_stats(),
        "failed_checks": analyze_failed_checks(),
        "llm_queries": analyze_llm_queries(),
        "llm_responses": analyze_llm_responses(),
        "evaluation": analyze_evaluation()
    }
    
    # Generate overall summary
    generate_pipeline_summary(results)
    
    logger.info("All analyses completed successfully")

if __name__ == "__main__":
    main()