import os
import sys
import json
import yaml
import logging
import argparse
import requests
import pandas as pd
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import subprocess

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class Config:
    artifact_hub_url: str = "https://artifacthub.io/api/v1/helm-exporter"
    templates_dir: str = "templates"
    results_dir: str = "results"
    tools_output_dir: str = "tools_output"
    log_file: Optional[str] = None
    log_level: str = "INFO"
    max_workers: int = os.cpu_count() or 1
    charts_limit: Optional[int] = None

def setup_logging(config: Config) -> None:
    """Configure logging handlers and format"""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger.setLevel(config.log_level)

    if config.log_file:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def create_directories(config: Config) -> None:
    """Create required directory structure"""
    os.makedirs(config.templates_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(os.path.join(config.tools_output_dir, "checkov"), exist_ok=True)
    os.makedirs(os.path.join(config.tools_output_dir, "datree"), exist_ok=True)

def fetch_helm_charts(config: Config) -> List[Dict]:
    """Fetch all Helm charts from Artifact Hub"""
    
    try:
        response = requests.get(config.artifact_hub_url, timeout=30)
        response.raise_for_status()
        data = response.json()
                
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch charts: {e}")
        sys.exit(1)
        
    return data

def process_charts(data: Dict, config: Config) -> Tuple[bool, str]:
    """Process the retrieved Helm charts to prodcue templates"""
    correct_templates = []
    failed_templates = []
    
    for chart in data:
        name = chart["name"]
        url = chart["repository"]["url"]
        try:
            # Add Helm repository
            subprocess.run(
                ["helm", "repo", "add", name, url],
                check = True,
                capture_output = True,
                text = True
            )
        except subprocess.CalledProcessError as e:
            return False, f"Repo add failed: {e.stderr}"

    # Generate template
    try:
        output_path = os.path.join(config.templates_dir, f"{name}_template.yaml")
        cmd = f"helm template {name}/{name} > {output_path}"
        output = subprocess.getoutput(cmd)
        
        if "Error" in output:
            error = {
                "chart": name,
                "error": output
            }
            failed_templates.append(error)
            os.system(f"rm {output_path}")
        else:
            correct_templates.append(name)
    except subprocess.CalledProcessError as e:
        return False, f"Template failed: {e.stderr}"


def generate_templates(config: Config) -> Tuple[List[str], List[Dict]]:
    """Generate Helm templates for all charts"""
    charts = fetch_helm_charts(config)
    unique_repos = {(c['repository']['name'], c['repository']['url']) for c in charts}
    
    # Add all repositories first
    for repo_name, repo_url in unique_repos:
        try:
            subprocess.run(
                ["helm", "repo", "add", repo_name, repo_url],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to add repo {repo_name}: {e.stderr}")

    # Update repositories
    try:
        subprocess.run(
            ["helm", "repo", "update"],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Repo update failed: {e.stderr}")

    # Process charts
    success = []
    failed = []
    
    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        futures = []
        for chart in charts:
            repo_name = chart['repository']['name']
            chart_name = chart['name']
            repo_url = chart['repository']['url']
            futures.append(
                executor.submit(
                    process_chart,
                    repo_name,
                    repo_url,
                    chart_name,
                    config
                )
            )

        for future in as_completed(futures):
            status, message = future.result()
            if status:
                success.append(message)
            else:
                failed.append(message)

    return success, failed

def analyze_template(file_path: str) -> Dict:
    """Analyze a single template file"""
    result = {
        "chart": os.path.basename(file_path).replace("_template.yaml", ""),
        "status": "Correct",
        "lines": 0,
        "containers": 0,
        "characters": 0
    }
    
    try:
        with open(file_path, "r") as f:
            content = f.read()
            result["lines"] = len(content.splitlines())
            result["characters"] = len(content)
            
            if result["lines"] == 0:
                result["status"] = "Empty"
                return result

            # Parse YAML
            docs = list(yaml.safe_load_all(content))
            docs = [d for d in docs if d and d.get("kind") != "PodSecurityPolicy"]
            
            if not docs:
                result["status"] = "Empty"
                return result
            
            print(docs)
            
            # Count containers
            containers = 0
            for doc in docs:
                spec = doc.get("spec", {})
                if "containers" in spec:
                    containers += len(spec["containers"])
                elif "template" in spec:
                    template_spec = spec.get("template", {}).get("spec", {})
                    containers += len(template_spec.get("containers", []))
                    
            result["containers"] = containers
            
    except yaml.YAMLError as e:
        result["status"] = f"YAML Error: {str(e)}"
    except Exception as e:
        result["status"] = f"Error: {str(e)}"
        
    return result

def generate_template_stats(config: Config) -> pd.DataFrame:
    """Generate statistics for all templates"""
    templates = [
        os.path.join(config.templates_dir, f)
        for f in os.listdir(config.templates_dir)
        if f.endswith("_template.yaml")
    ]
    
    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        results = list(executor.map(analyze_template, templates))
    
    df = pd.DataFrame(results)
    stats_path = os.path.join(config.results_dir, "template_stats.csv")
    df.to_csv(stats_path, index=False)
    return df

def run_tool(tool_name: str, chart: str, config: Config) -> bool:
    """Run a static analysis tool on a chart"""
    template_path = os.path.join(config.templates_dir, f"{chart}_template.yaml")
    output_dir = os.path.join(config.tools_output_dir, tool_name)
    
    if tool_name == "checkov":
        cmd = [
            "checkov",
            "-f", template_path,
            "--quiet",
            "--compact",
            "--framework", "kubernetes",
            "-o", "json"
        ]
    elif tool_name == "datree":
        cmd = [
            "helm", "datree", "test",
            template_path,
            "--only-k8s-files",
            "--quiet",
            "--output", "json"
        ]
    else:
        return False

    output_path = os.path.join(output_dir, f"{chart}_results.json")
    
    try:
        with open(output_path, "w") as f:
            subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{tool_name} failed for {chart}: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error running {tool_name} on {chart}: {str(e)}")
        return False

def run_static_analysis(config: Config, charts: List[str]) -> None:
    """Run static analysis tools in parallel"""
    tools = ["checkov", "datree"]
    
    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        for tool in tools:
            futures = [
                executor.submit(run_tool, tool, chart, config)
                for chart in charts
            ]
            for future in as_completed(futures):
                future.result()

def main():
    parser = argparse.ArgumentParser(description="Helm Chart Analyzer")
    parser.add_argument("--log-file", help="Path to log file")
    parser.add_argument("--log-level", default="INFO", 
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--max-workers", type=int, default=os.cpu_count())
    parser.add_argument("--charts-limit", type=int, help="Limit number of charts to process")
    args = parser.parse_args()

    config = Config(
        log_file=args.log_file,
        log_level=args.log_level,
        max_workers=args.max_workers,
        charts_limit=args.charts_limit
    )
    
    setup_logging(config)
    create_directories(config)
    
    logger.info("Fetching Helm charts...")
    success, failed = generate_templates(config)
    logger.info(f"Generated {len(success)} templates, failed {len(failed)}")
    
    logger.info("Analyzing templates...")
    stats_df = generate_template_stats(config)
    valid_charts = stats_df[stats_df["status"] == "Correct"]["chart"].tolist()
    
    logger.info("Running static analysis...")
    run_static_analysis(config, valid_charts)
    
    logger.info("Processing complete")

if __name__ == "__main__":
    main()