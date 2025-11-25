import importlib
import sys
import os
from typing import Any, Type

def load_class(module_path: str, class_name: str) -> Type[Any]:
    """
    Dynamically loads a class from a module path.
    
    Args:
        module_path: Dot-separated path to the module (e.g., 'src.plugins.my_plugin')
        class_name: Name of the class to load (e.g., 'MyCustomLLM')
        
    Returns:
        The class type
        
    Raises:
        ImportError: If module cannot be imported
        AttributeError: If class is not found in module
    """
    try:
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
            
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_path}': {e}")
    except AttributeError:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_path}'")

def instantiate_class(module_path: str, class_name: str, **kwargs) -> Any:
    """
    Dynamically instantiates a class.
    
    Args:
        module_path: Dot-separated path to the module
        class_name: Name of the class to load
        **kwargs: Arguments to pass to the class constructor
        
    Returns:
        An instance of the class
    """
    cls = load_class(module_path, class_name)
    return cls(**kwargs)
