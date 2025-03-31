#!/bin/bash

# Run setup script if specified
if [[ "$1" == "setup" ]]; then
    echo "Running setup script..."
    ./shell-scripts/setup.sh
fi

# Activate virtual environment
echo "Activating virtual environment..."
. .venv/bin/activate

# Run python scripts
echo "Running chart retrieval..."
python3 ./src/chart_retrieval.py --log-file experiment2/logs/chart_retrieval.log --charts-limit 20

echo "Running tool parsing..."
python3 src/parse_tools.py --log-file experiment2/logs/tool_parsing.log

echo "Querying LLM..."
python3 src/querry_llm.py --log-file experiment2/logs/llm_querrying.log

echo "Evaluating queries..."
python3 src/validate_llm.py --log-file experiment2/logs/llm_evaluation.log

echo "Generating stats..."
python3 src/stats.py --log-file experiment2/logs/stats.log --output-dir experiment2/stats 

echo "Running second analysis..."
python3 src/evaluate_fixes.py --log-file experiment2/logs/fixes_validation.log