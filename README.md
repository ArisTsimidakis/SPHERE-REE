# LLM Helm Fix REE
This repository contains my REE submission for the SPHERE summer internship.
It focuses on automating and making it as easy as possible to replicate the research conducted on the paper ["Analyzing and Mitigating (with LLMs) the Security Misconfigurations of Helm Charts from Artifact Hub"](https://arxiv.org/abs/2403.09537).

## Setup and execution
All steps required for setting up your environment and executing the pipeline have been automated and are accessible through shell scripts.
You can run
```sh
./shell-scripts/setup.sh
./pipeline.sh
```
or alternatively, 
```sh
./pipeline.sh setup
```
In order to change the arguents passed to each script, you can directly change them from the ```pipeline.sh``` script. Otherwise, run each script individually, specifying your desired arguments. In order to change the directory structure/IO files etc, you can modify the fields in the config class of each script. To see the available arguments for each script:
```sh
python3 src/script.py -h
```

## Pipeline description
The pipeline consists of the following scripts:
### 1.  chart_retrieval.py
- Fetches all helm charts from artifact hub and performs static analysis, saving the output in json files.

### 2. parse_tools.py
- Parses the output of the static analysis tools for each helm chart, and generates a csv containing information about each reported error, which chart it comes from, from which tool it was reported, and the relevant yaml snippet that causes the error.
- Generates a csv containing the query to send to the LLM for each reported error.

### 3. query_llm.py
- Uses LLM APIs to retrieve fixes for the reported misconfigurations. Saves both the fixed snippet and original snippet in a csv file.

### 4. validate_llm.py
- Uses kubeconform to validate the YAML syntax of the snippet provided by the LLM. Reports on statistics such as added/changed/removed lines, and syntax improvement.

### 5. stats.py
- Creates general statistics and graphs about each step of the pipeline.

### 6. evaluate_fixes.py
- Re-runs the static analysis tools on the fixed snippets provided by the LLM, and evaluates how many misconfigurations are still being reported (i.e. were not fixed). Reports a variety of statistics about the results. 


## Supported static analysis tools and LLMs
As of now, the only supported static analysis tools are checkov and datree.
The only supported LLM is ChatGPT.

However, the code is designed to be very easy to maintain and integrate new features, so integration of more tools and LLMs will be easy and seamless.