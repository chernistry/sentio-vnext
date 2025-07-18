"""
Dependency checker for Sentio vNext.

This module checks for the presence of required dependencies and
provides helpful error messages when they're missing.
"""

import importlib.util
import logging
import sys
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Required dependencies by component
DEPENDENCIES: Dict[str, List[str]] = {
    "core": ["httpx", "requests"],
    "embeddings": ["httpx", "requests"],
    "ingest": ["pandas", "typer"],
    "api": ["fastapi", "uvicorn", "pydantic"],
    "vector_store": ["qdrant-client"],
    "rerank": ["sentence-transformers"],
}


def is_package_installed(package_name: str) -> bool:
    """
    Check if a package is installed.
    
    Args:
        package_name: Name of the package to check.
        
    Returns:
        True if installed, False otherwise.
    """
    return importlib.util.find_spec(package_name) is not None


def check_dependencies(components: Optional[List[str]] = None) -> Dict[str, Set[str]]:
    """
    Check if all required dependencies are installed.
    
    Args:
        components: List of component names to check. If None, checks all.
        
    Returns:
        Dictionary of {component_name: missing_deps}
    """
    components = components or list(DEPENDENCIES.keys())
    
    missing_deps: Dict[str, Set[str]] = {}
    
    for component in components:
        if component not in DEPENDENCIES:
            logger.warning(f"Unknown component: {component}")
            continue
            
        missing = {
            dep for dep in DEPENDENCIES[component]
            if not is_package_installed(dep)
        }
        
        if missing:
            missing_deps[component] = missing
    
    return missing_deps


def print_dependency_report(missing_deps: Dict[str, Set[str]]) -> bool:
    """
    Print a report of missing dependencies.
    
    Args:
        missing_deps: Dictionary of {component_name: missing_deps}.
        
    Returns:
        True if all dependencies are installed, False otherwise.
    """
    if not missing_deps:
        print("✅ All dependencies installed!")
        return True
        
    print("❌ Missing dependencies:")
    for component, deps in missing_deps.items():
        print(f"  - {component}: {', '.join(deps)}")
    
    # Print installation instructions
    all_missing = set()
    for deps in missing_deps.values():
        all_missing.update(deps)
    
    if all_missing:
        print("\nInstall missing dependencies with:")
        print(f"pip install {' '.join(all_missing)}")
    
    return False


if __name__ == "__main__":
    missing = check_dependencies()
    if not print_dependency_report(missing):
        sys.exit(1) 