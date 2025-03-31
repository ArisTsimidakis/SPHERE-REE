import csv
import json
import logging
import argparse
import os
from helpers import *
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)
config = None 

# Specify directories for reference
script_dir = Path(__file__).parent.resolve()
root_dir = script_dir.parent

@dataclass
class Config:
    # Directories
    templates_dir = os.path.join(root_dir, "experiment2", "templates")
    results_dir = os.path.join(root_dir, "experiment2", "results")
    tools_output_dir = os.path.join(root_dir, "experiment2", "tools_output")

    # Logging
    log_file: Optional[str] = None
    log_level: Optional[str] = "INFO"

    max_workers: Optional[int] = os.cpu_count() or 1

def parse_args():
    parser = argparse.ArgumentParser(description="Process Kubernetes scanning tool results.")
    parser.add_argument("--log-file", help="Path to log file", default=None)
    parser.add_argument("--log-level", help="Logging level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
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
            
            # When log file is specified, don't add console handler
            return
        except Exception as e:
            # If file logging fails, fall back to console
            logger.error(f"Failed to setup file logging: {str(e)}")
    
    # Console handler 
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(config.log_level.upper())
    logger.addHandler(console_handler)

    logger.info("Logging initialized with level: %s", config.log_level)

def save_to_csv(data: List[Union[Dict, List]], filename: str, fieldnames: List[str] = None) -> None:
    """Save data to a CSV file."""
    try:
        logger.debug(f"Attempting to save data to CSV: {filename}")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            if isinstance(data[0], dict):
                writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            else:
                writer = csv.writer(csvfile)
                if fieldnames:
                    writer.writerow(fieldnames)
                writer.writerows(data)
        logger.info(f"Successfully saved {len(data)} rows to {filename}")
    except Exception as e:
        logger.error(f"Failed to save CSV {filename}: {str(e)}")
        raise

def process_tool_results(tool_name: str, chart_name: str) -> List[Union[Dict, List]]:
    """Process results for a single tool and chart."""
    logger.debug(f"Processing {tool_name} results for chart: {chart_name}")
    try:
        if tool_name == "checkov":
            result_file = f"{config.tools_output_dir}/checkov/{chart_name}_results.json"
            logger.debug(f"Processing Checkov file: {result_file}")
            return parse_checkov(chart_name, result_file)
        elif tool_name == "datree":
            result_file = f"{config.tools_output_dir}/datree/{chart_name}_results.json"
            logger.debug(f"Processing Datree file: {result_file}")
            return parse_datree(chart_name)
        else:
            logger.warning(f"Unknown tool: {tool_name}")
            return []
    except Exception as e:
        logger.error(f"Error processing {tool_name} results for {chart_name}: {str(e)}", exc_info=True)
        return []

# CHECKOV functions with added logging
def get_ckv_container_objects(template: list, resource_path: str, container_type: str = "containers") -> list:
    """Returns the container objects based on the resource path."""
    logger.debug(f"Getting container objects for {resource_path} (type: {container_type})")
    for document in template:
        if not check_resource_path(resource_path.split("/"), document):
            continue

        spec = document.get("spec", {})
        if "template" in spec:
            spec = spec["template"].get("spec", {})

        containers = spec.get(container_type, [])
        logger.debug(f"Found {len(containers)} {container_type} for {resource_path}")
        return containers
    
    logger.debug(f"No containers found for {resource_path}")
    return []

def get_ckv_resource_path(check: dict, template: list) -> dict:
    """Returns the K8s resource path where there is the misconfiguration."""
    logger.debug(f"Getting resource path for check: {check.get('check_id')}")
    paths = {"resource_path": "", "obj_path": ""}
    resource_parts = check["resource"].split(".")[:3]
    paths["resource_path"] = "/".join(resource_parts)

    if check["check_id"] == "CKV2_K8S_5":
        return paths

    evaluated_keys = check["check_result"].get("evaluated_keys", [])
    if not evaluated_keys:
        logger.debug("No evaluated keys found in check")
        return paths

    obj_path = evaluated_keys[0]
    index = obj_path.rfind("]/")
    if index != -1:
        obj_path = obj_path[:index+2]
    
    obj_path = obj_path.replace("[", "").replace("]", "").rstrip("/")
    paths["obj_path"] = obj_path

    container_types = []
    if "initContainers" in obj_path:
        container_types.append("initContainers")
    if "containers" in obj_path:
        container_types.append("containers")

    for c_type in container_types:
        containers = get_ckv_container_objects(template, paths["resource_path"], c_type)
        for idx in range(len(containers)):
            modified_path = obj_path.replace("containers", c_type)[:-1] + str(idx)
            paths["obj_path"] = modified_path
            logger.debug(f"Modified path for container: {modified_path}")

    logger.debug(f"Final paths: {paths}")
    return paths

def parse_checkov(chart_name: str, path: str) -> list[dict]:
    """Parse the output of the Checkov tool."""
    logger.info(f"Parsing Checkov results for {chart_name} from {path}")
    try:
        with open(path, 'r', encoding="utf-8") as file:
            results = json.load(file)
        logger.debug(f"Successfully loaded Checkov JSON for {chart_name}")
    except json.decoder.JSONDecodeError as e:
        logger.error(f"Error decoding JSON for {chart_name}: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading Checkov results for {chart_name}: {str(e)}")
        return []

    if not results.get("results", {}).get("failed_checks"):
        logger.info(f"No failed checks found in Checkov results for {chart_name}")
        return []

    filepath = f"{config.templates_dir}/{chart_name}_template.yaml"
    logger.debug(f"Loading template from {filepath}")
    template = parse_yaml_template(filepath)
    if not template:
        logger.warning(f"Empty template for {chart_name}")
        return []

    rows = []
    logger.info(f"Processing {len(results['results']['failed_checks'])} failed checks for {chart_name}")

    for check in results["results"]["failed_checks"]:
        logger.debug(f"Processing check: {check['check_id']}")
        paths = get_ckv_resource_path(check, template)
        yaml_snippet = get_resource_snippet(paths, template)
        
        std_check_id = CheckovLookup.get_value(check['check_id']) or "NOT_MAPPED"
        cis_cluster = CISLookup.get_value(std_check_id) or "N/A"
        
        snippet_length = len(str(yaml_snippet))
        if snippet_length < 10:
            template_str = "".join(str(doc) for doc in template)
            snippet_length = len(template_str)
            logger.debug(f"Using full template as snippet was too small ({snippet_length} chars)")

        rows.append({
            "Chart": chart_name,
            "Tool": "checkov",
            "Alert_ID": check['check_id'],
            "Standard_ID": std_check_id,
            "CIS_Cluster": cis_cluster,
            "Description": check['check_name'],
            "Resource": [paths["resource_path"], paths["obj_path"]],
            "Original_YAML": yaml_snippet,
            "Characters": snippet_length
        })

    logger.info(f"Processed {len(rows)} rows for {chart_name}")
    return rows

# DATREE functions with added logging
def get_datree_path(occurrence: dict) -> dict:
    """Get the path of the object that caused the failure in Datree results."""
    logger.debug(f"Getting path for Datree occurrence: {occurrence.get('kind')}")
    resource_path = f"{occurrence['kind']}/{occurrence['metadataName']}"
    obj_path = ""

    if occurrence.get("failureLocations"):
        failure = occurrence["failureLocations"][0]
        obj_path = failure.get("schemaPath", "")

        if obj_path:
            # Handle paths with digits (array indices)
            if any(c.isdigit() for c in obj_path):
                for idx, char in enumerate(obj_path):
                    if char.isdigit():
                        obj_path = obj_path[:idx+1]
                        break
            else:
                # Handle regular paths by removing last component
                obj_path = obj_path.rsplit('/', 1)[0]

    logger.debug(f"Datree paths - resource: {resource_path}, object: {obj_path}")
    return {
        "resource_path": resource_path,
        "obj_path": obj_path
    }

def parse_datree(chart_name: str) -> list[dict]:
    """Parse the output of the Datree tool for a given chart."""
    json_path = f"{config.tools_output_dir}/datree/{chart_name}_results.json"
    logger.info(f"Parsing Datree results for {chart_name} from {json_path}")

    try:
        with open(json_path, 'r', encoding="utf-8") as file:
            results = json.load(file)
        logger.debug(f"Successfully loaded Datree JSON for {chart_name}")
    except json.decoder.JSONDecodeError as e:
        logger.error(f"Error decoding JSON for {chart_name}: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading Datree results for {chart_name}: {str(e)}")
        return []

    policy_results = results.get("policyValidationResults", [])
    if not policy_results or not policy_results[0].get("ruleResults"):
        logger.info(f"No policy results found in Datree output for {chart_name}")
        return []

    rows = []

    filepath = f"{config.templates_dir}/{chart_name}_template.yaml"
    logger.debug(f"Loading template from {filepath}")
    template = parse_yaml_template(filepath)
    if not template:
        logger.warning(f"Empty template for {chart_name}")
        return []

    logger.info(f"Processing {len(policy_results[0]['ruleResults'])} rule results for {chart_name}")

    for check in policy_results[0]["ruleResults"]:
        check_id = check['identifier']
        descr = check['name']
        logger.debug(f"Processing check: {check_id} - {descr}")
        
        std_check_id = DatreeLookup.get_value(check_id) or "NOT_MAPPED"
        cis_cluster = CISLookup.get_value(std_check_id) or "N/A"

        for occurrence in check.get("occurrencesDetails", []):
            paths = get_datree_path(occurrence)
            yaml_snippet = get_resource_snippet(paths, template)
            
            snippet_length = len(str(yaml_snippet))
            if snippet_length < 10:
                template_str = "".join(str(doc) for doc in template)
                snippet_length = len(template_str)
                logger.debug(f"Using full template as snippet was too small ({snippet_length} chars)")

            rows.append({
                "Chart": chart_name,
                "Tool": "datree",
                "Alert_ID": check_id,
                "Standard_ID": std_check_id,
                "CIS_Cluster": cis_cluster,
                "Description": descr,
                "Resource": [paths["resource_path"], paths["obj_path"]],
                "Original_YAML": yaml_snippet,
                "Characters": snippet_length
            })

    logger.info(f"Processed {len(rows)} rows for {chart_name}")
    return rows

def build_queries(rows):
    """Build the query for each row."""
    logger.info(f"Building queries from {len(rows)} rows")
    query_data = []

    for row in rows:
        resource = row["Resource"] 
        paths = {
            "resource_path": resource[0],
            "obj_path": resource[1]
        }
        resource_type = paths["resource_path"].split("/")[0]

        query = "You are a software engineer working on a Kubernetes project. You need to refactor the following " + resource_type + " YAML resource because " + row["Description"].lower() + ". You must only generate YAML code between --- characters, with no additional text or description."

        if row["Original_YAML"]:
            query_data.append({
                "Chart": row["Chart"],
                "Alert_ID": row["Alert_ID"],
                "Tool": row["Tool"],
                "Resource": row["Resource"],
                "Query": query,
                "Original_YAML": row["Original_YAML"]
            })
            logger.debug(f"Built query for {row['Chart']} - {row['Alert_ID']}")
        else:
            logger.warning(f"Skipping query for {row['Chart']} - {row['Alert_ID']} due to missing YAML")
    
    logger.info(f"Built {len(query_data)} queries")
    return query_data

def main():
    """Main function to process all tool results and generate output files."""
    global config
    
    args = parse_args()
    config = Config(log_file=args.log_file, log_level=args.log_level)
    
    # Setup logging after config is initialized
    setup_logging()
    
    logger.info("Starting processing of tool results")
    logger.debug(f"Configuration: {config}")

    # Define tools and output files
    tools = ["checkov", "datree"]
    output_dir = config.results_dir
    all_failed_checks = []
    all_queries = []
    
    # Get all chart names by scanning tool directories
    chart_names = set()
    for tool in tools:
        tool_dir = Path(f"{config.tools_output_dir}/{tool}")
        if tool_dir.exists():
            logger.debug(f"Scanning tool directory: {tool_dir}")
            for result_file in tool_dir.glob("*.json"):
                chart_name = result_file.stem.replace("_results", "")
                chart_names.add(chart_name)
                logger.debug(f"Found chart: {chart_name} in {tool} results")

    if not chart_names:
        logger.warning("No chart results found in tools_output directories")
        return

    logger.info(f"Found {len(chart_names)} charts to process: {chart_names}")

    # Process all charts 
    with ThreadPoolExecutor(max_workers = config.max_workers) as executor:
        futures = []
        for tool in tools:
            for chart_name in chart_names:
                logger.debug(f"Submitting task for {tool}/{chart_name}")
                futures.append(
                    executor.submit(process_tool_results, tool, chart_name)
                )

        # Process results as they complete
        logger.info(f"Processing {len(futures)} tasks with {config.max_workers} workers")
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tools"):
            try:
                result = future.result()
                if result:
                    all_failed_checks.extend(result)
                    logger.debug(f"Processed task with {len(result)} results")
            except Exception as e:
                logger.error(f"Error processing future: {str(e)}", exc_info=True)

    if not all_failed_checks:
        logger.warning("No failed checks found in any tool results")
        return

    logger.info(f"Total failed checks found: {len(all_failed_checks)}")

    # Save failed checks to CSV
    failed_checks_csv = f"{output_dir}/failed_checks.csv"
    logger.info(f"Saving failed checks to {failed_checks_csv}")
    save_to_csv(all_failed_checks, failed_checks_csv)

    # Build and save queries
    logger.info("Building queries from failed checks")
    queries = build_queries(all_failed_checks)
    queries_csv = f"{output_dir}/llm_queries.csv"
    query_fieldnames = [
        "Chart", "Alert_ID", "Tool", "Resource", "Query", "Original_YAML"
    ]
    logger.info(f"Saving {len(queries)} queries to {queries_csv}")
    save_to_csv(queries, queries_csv, query_fieldnames)

    logger.info("Processing completed successfully")

if __name__ == "__main__":
    main()