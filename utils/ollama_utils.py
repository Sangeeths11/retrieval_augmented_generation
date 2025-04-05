import requests
import sys
from typing import List, Dict, Any, Optional

def check_ollama(base_url: str = "http://localhost:11434") -> bool:
    """
    Check if Ollama is running.
    
    Args:
        base_url: The base URL for Ollama
        
    Returns:
        True if Ollama is running, False otherwise
    """
    try:
        response = requests.get(f"{base_url}/api/version")
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def list_models(base_url: str = "http://localhost:11434") -> List[str]:
    """
    List models available in Ollama.
    
    Args:
        base_url: The base URL for Ollama
        
    Returns:
        List of model names
    """
    try:
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model.get("name") for model in models]
        return []
    except requests.exceptions.RequestException:
        return []

def check_required_models(required_models: List[str], base_url: str = "http://localhost:11434") -> Dict[str, bool]:
    """
    Check if required models are available in Ollama.
    
    Args:
        required_models: List of required model names
        base_url: The base URL for Ollama
        
    Returns:
        Dictionary mapping model names to availability
    """
    available_models = list_models(base_url)
    return {model: model in available_models for model in required_models}

def model_info(model_name: str, base_url: str = "http://localhost:11434") -> Optional[Dict[str, Any]]:
    """
    Get information about a model.
    
    Args:
        model_name: The name of the model
        base_url: The base URL for Ollama
        
    Returns:
        Model information or None if unavailable
    """
    try:
        response = requests.post(
            f"{base_url}/api/show", 
            json={"name": model_name}
        )
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None

def print_ollama_status(required_models: List[str] = None, base_url: str = "http://localhost:11434"):
    """
    Print Ollama status and model availability.
    
    Args:
        required_models: List of required model names
        base_url: The base URL for Ollama
    """
    if required_models is None:
        required_models = ["gemma3:12b", "nomic-embed-text:latest"]
    
    print("\n== Ollama Status ==")
    
    # Check if Ollama is running
    if not check_ollama(base_url):
        print("❌ Ollama is not running. Please start Ollama.")
        print(f"   Expected at: {base_url}")
        return
    
    print("✅ Ollama is running")
    
    # Check required models
    model_status = check_required_models(required_models, base_url)
    
    print("\nRequired models:")
    for model, available in model_status.items():
        if available:
            print(f"✅ {model} is available")
        else:
            print(f"❌ {model} is not available. Pull with: ollama pull {model}")
    
    # List all available models
    all_models = list_models(base_url)
    
    print(f"\nAll available models ({len(all_models)}):")
    for model in all_models:
        print(f"  - {model}")
    
    print("\nTo pull missing models:")
    missing_models = [model for model, available in model_status.items() if not available]
    if missing_models:
        for model in missing_models:
            print(f"  ollama pull {model}")
    else:
        print("  All required models are available!")
    
    print("")

if __name__ == "__main__":
    # If run directly, print Ollama status
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
    print_ollama_status(base_url=base_url) 