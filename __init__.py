import os
import sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

#from .launch import prepare_environment
#prepare_environment()

#from .launch import download_models
#download_models()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
