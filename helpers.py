from typing import Callable
import yaml
import json
import os


# This file defines the following classes and functions:
# - CISLookup
# - CheckovLookup
# - parse_yaml_template
# - check_resource_path
# - get_resource_snippet
# - save_yaml_snippet
# - get_ckv_container_objects
# - get_ckv_resource_path
# - parse_checkov
# - build_queries


################################################################
#                  Lookup class definitions                    #
################################################################
class CISLookup:
    """This class is used to lookup the CIS Benchmark cluster ID.
    """

    _LOOKUP = {
        "check_10": "Pod Security Policy",
        "check_11": "Pod Security Policy",
        "check_12": "Pod Security Policy",
        "check_21": "Pod Security Policy",
        "check_22": "Pod Security Policy",
        "check_23": "Pod Security Policy",
        "check_24": "Pod Security Policy",
        "check_26": "General Policies",
        "check_28": "Pod Security Policy",
        "check_30": "General Policies",
        "check_31": "General Policies",
        "check_33": "Secrets Management",
        "check_34": "Pod Security Policy",
        "check_35": "RBAC and Service Accounts",
        "check_36": "RBAC and Service Accounts",
        "check_37": "RBAC and Service Accounts",
        "check_39": "RBAC and Service Accounts",
        "check_40": "Network Policies and CNI",
        "check_54": "RBAC and Service Accounts",
        "check_59": "RBAC and Service Accounts",
        "check_70": "Network Policies and CNI",
        "check_77": "RBAC and Service Accounts",
        "check_78": "RBAC and Service Accounts",
        "check_80": "RBAC and Service Accounts"
    }

    @classmethod
    def get_value(cls, key) -> Callable:
        return cls._LOOKUP.get(key)



class CheckovLookup:
    """This class is used to lookup the function to be called for each check.
    """

    _LOOKUP = {
        "CKV_K8S_1": "",
        "CKV_K8S_2": "",
        "CKV_K8S_3": "",
        "CKV_K8S_4": "",
        "CKV_K8S_5": "",
        "CKV_K8S_6": "",
        "CKV_K8S_7": "",
        "CKV_K8S_8": "check_7", 
        "CKV_K8S_9": "check_8", 
        "CKV_K8S_10": "check_4", 
        "CKV_K8S_11": "check_5", 
        "CKV_K8S_12": "check_1", 
        "CKV_K8S_13": "check_2", 
        "CKV_K8S_14": "check_0", 
        "CKV_K8S_15": "check_25", 
        "CKV_K8S_16": "check_21", 
        "CKV_K8S_17": "check_10", 
        "CKV_K8S_18": "check_11", 
        "CKV_K8S_19": "check_12",
        "CKV_K8S_20": "check_22", 
        "CKV_K8S_21": "check_26", 
        "CKV_K8S_22": "check_27", 
        "CKV_K8S_23": "check_28", 
        "CKV_K8S_25": "check_34", 
        "CKV_K8S_26": "check_29",
        "CKV_K8S_28": "check_34", 
        "CKV_K8S_29": "check_30", 
        "CKV_K8S_31": "check_31", 
        "CKV_K8S_32": "check_31",
        "CKV_K8S_33": "check_38",
        "CKV_K8S_35": "check_33", 
        "CKV_K8S_36": "check_34",
        "CKV_K8S_24": "check_34",
        "CKV_K8S_37": "check_34", 
        "CKV_K8S_38": "check_35", 
        "CKV_K8S_39": "check_34", 
        "CKV_K8S_40": "check_13", 
        "CKV_K8S_41": "check_35", 
        "CKV_K8S_42": "check_35", 
        "CKV_K8S_43": "check_9", 
        "CKV2_K8S_6": "check_40",
        "CKV_K8S_30": "check_30",
        "CKV_K8S_155": "check_54",
        "CKV2_K8S_5": "check_59",
        "CKV_K8S_49": "check_54",
        "CKV_K8S_156": "check_54",
        "CKV_K8S_157": "check_54",
        "CKV_K8S_158": "check_54",
        "CKV2_K8S_3": "check_54",
        "CKV2_K8S_4": "check_54",
        "CKV2_K8S_2": "check_54",
        "CKV_K8S_27": "check_15",
        "CKV2_K8S_1": "check_54",
        "CKV_K8S_153": "check_73",
        "CKV_K8S_119": "check_74",
        "CKV_K8S_116": "check_74",
        "CKV_K8S_117": "check_75",
        "CKV_K8S_34": "check_72",
    }

    @classmethod
    def get_value(cls, key) -> Callable:
        """ Get the function to be called for each check.

        Args:
            key (str): The check number.
        """
        return cls._LOOKUP.get(key)

    @classmethod
    def print_value(cls, key) -> None:
        """ Print the function to be called for each check."""
        print(cls._LOOKUP.get(key))
        
        
class DatreeLookup:
    """This class is used to lookup the function to be called for each check.
    """

    _LOOKUP = {
        "CONTAINERS_MISSING_LIVENESSPROBE_KEY": "check_7",
        "CONTAINERS_MISSING_READINESSPROBE_KEY": "check_8",
        "CONTAINERS_MISSING_CPU_REQUEST_KEY": "check_4",
        "CONTAINERS_MISSING_CPU_LIMIT_KEY": "check_5",
        "CONTAINERS_MISSING_MEMORY_REQUEST_KEY": "check_1",
        "CONTAINERS_MISSING_MEMORY_LIMIT_KEY": "check_2",
        "CONTAINERS_MISSING_IMAGE_VALUE_VERSION": "check_0",
        "CONTAINERS_INCORRECT_PRIVILEGED_VALUE_TRUE": "check_21",
        "CONTAINERS_INCORRECT_HOSTPID_VALUE_TRUE": "check_10",
        "CONTAINERS_INCORRECT_HOSTIPC_VALUE_TRUE": "check_11",
        "CONTAINERS_INCORRECT_HOSTNETWORK_VALUE_TRUE": "check_12",
        "CONTAINERS_MISSING_KEY_ALLOWPRIVILEGEESCALATION": "check_22",
        "WORKLOAD_INCORRECT_NAMESPACE_VALUE_DEFAULT": "check_26",
        "CONTAINERS_INCORRECT_READONLYROOTFILESYSTEM_VALUE": "check_27",
        "CONTAINERS_INCORRECT_RUNASNONROOT_VALUE": "check_28",
        "CIS_MISSING_KEY_SECURITYCONTEXT": "check_30",
        "CONTAINERS_INCORRECT_SECCOMP_PROFILE": "check_31",
        "CIS_INVALID_VALUE_SECCOMP_PROFILE": "check_31",
        "CONTAINERS_INVALID_CAPABILITIES_VALUE": "check_23",
        "CIS_MISSING_VALUE_DROP_NET_RAW": "check_23",
        "CIS_INVALID_VALUE_AUTOMOUNTSERVICEACCOUNTTOKEN": "check_35",
        "SRVACC_INCORRECT_AUTOMOUNTSERVICEACCOUNTTOKEN_VALUE": "check_35",
        "CONTAINERS_MISSING_IMAGE_VALUE_DIGEST": "check_9",
        "CIS_INVALID_KEY_SECRETKEYREF_SECRETREF": "check_33",
        "DEPLOYMENT_INCORRECT_REPLICAS_VALUE": "check_45",
        "SERVICE_INCORRECT_TYPE_VALUE_NODEPORT": "check_56",
        "CONTAINERS_INCORRECT_RUNASUSER_VALUE_LOWUID": "check_13",
        "CONTAINERS_INCORRECT_KEY_HOSTPORT": "check_29",
        "CONTAINER_CVE2021_25741_INCORRECT_SUBPATH_KEY": "check_50",
        "CIS_INVALID_VERB_SECRETS": "check_54",
        "CONTAINERS_INCORRECT_RESOURCES_VERBS_VALUE": "check_54",
        "EKS_INVALID_CAPABILITIES_EKS": "check_34",
        "CIS_INVALID_VALUE_CREATE_POD": "check_54",
        "CIS_INVALID_WILDCARD_ROLE": "check_54",
        "CIS_INVALID_VALUE_BIND_IMPERSONATE_ESCALATE": "check_54",
        "CONTAINERS_INCORRECT_KEY_HOSTPATH": "check_47",
        "CIS_INVALID_ROLE_CLUSTER_ADMIN": "check_65",
        "INGRESS_INCORRECT_HOST_VALUE_PERMISSIVE": "check_66",
        "WORKLOAD_INVALID_LABELS_VALUE": "check_43",
    }

    @classmethod
    def get_value(cls, key) -> Callable:
        """ Get the function to be called for each check.

        Args:
            key (str): The check number.
        """
        return cls._LOOKUP.get(key)

    @classmethod
    def print_value(cls, key) -> None:
        """ Print the function to be called for each check."""
        print(cls._LOOKUP.get(key))

################################################################
#                  Helper function definitions                 #
################################################################

def parse_yaml_template(file_path: str) -> list:
    """Parses a Helm chart template yaml file and returns it as a list of dictionaries.

    Args:
        file_path: The path to the yaml file to parse.

    Returns:
        A list containing the parsed documents from the template.yaml file,
        filtered to remove null documents and PodSecurityPolicy kinds.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            template = list(yaml.load_all(file, Loader=yaml.FullLoader))
    except (yaml.scanner.ScannerError, yaml.constructor.ConstructorError) as e:
        print(f"Error parsing the YAML file: {e}")
        return []

    return [
        doc for doc in template 
        if doc is not None and doc.get("kind") != "PodSecurityPolicy"
    ]


def check_resource_path(path_list: str, document: dict) -> bool:
    """Check if the resource path exists in the template.

    Args:
        path_list: The resource path to check as a list of strings.
        document: The template document to check.

    Returns:
        True if the resource path exists, False otherwise.
    """
    if not path_list or not document:
        return False

    if document["kind"].casefold() != path_list[0].casefold():
        return False

    metadata = document.get("metadata", {})
    name = metadata.get("name")
    namespace = metadata.get("namespace")

    # Namespaces to ignore for name matching
    ignored_namespaces = {
        "default", 
        "test-ns", 
        "busybox-namespace", 
        "kube-system"
    }

    if namespace in ignored_namespaces:
        return name == path_list[-1]
    elif namespace == path_list[1]:
        return name == path_list[-1]
    elif not namespace and path_list[1] == "default":
        return name == path_list[-1]
    elif not namespace:
        return name == path_list[1]

    return False


def get_resource_snippet(paths: dict, template: list) -> dict:
    """Returns the resource snippet from the template based on the given paths.

    Args:
        paths: Dictionary containing 'resource_path' and 'obj_path'.
        template: List of parsed YAML documents.

    Returns:
        The requested resource snippet as a dictionary.
    """
    resource = {}
    resource_path = paths["resource_path"].split("/")
    obj_path = paths["obj_path"].split("/") if paths["obj_path"] else []

    for document in template:
        if not check_resource_path(resource_path, document):
            continue

        resource = document
        for key in obj_path:
            if not key or key == "env":
                break

            if key == "containers" and key not in resource:
                if "template" in resource:
                    resource = resource["template"]
                if "spec" in resource:
                    resource = resource["spec"]

            try:
                resource = resource[int(key)] if key.isdigit() else resource[key]
            except (TypeError, IndexError, KeyError):
                break

    return resource


def save_yaml_snippet(snippet: str | dict, snippet_type: str) -> None:
    """Save chart template data to a file.

    Args:
        snippet: The snippet data to be saved (either string or dict).
        snippet_type: Either 'original' or 'llm'.

    Raises:
        IOError: If there is an error writing to the file.
    """
    if not isinstance(snippet, dict):
        snippet = yaml.safe_load(snippet)

    file_path = f"tmp_snippets/{snippet_type}_snippet.yaml"
    with open(file_path, 'w', encoding="utf-8") as file:
        yaml.dump(snippet, file, sort_keys=False)
    
