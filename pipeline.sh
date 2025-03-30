#!/bin/bash

# Check if the first argument is 'setup'
if [[ "$1" == "setup" ]]; then
    echo "Running setup script..."
    ./setup.sh
fi

# Run the required Python scripts
echo "Running chart retrieval..."
python3 ./src/chart_retrieval.py --log-file logs/chart_retrieval.log

echo "Running tool parsing..."
python3 src/parse_tools.py --log-file logs/tool_parsing.log

echo "Querying LLM..."
python3 src/querry_llm.py --query --start-index 100 --end-index 200 --log-file logs/llm_querrying.log

echo "Evaluating queries..."
python3 src/querry_llm.py --evaluate --log-file logs/evaluation.log

echo "Generating stats..."
python3 src/querry_llm.py --stats --log-file stats.log