import sys, os
from pathlib import Path

# Adjust path to your project folder on PythonAnywhere
project_home = str(Path(__file__).parent.resolve())
if project_home not in sys.path:
    sys.path.insert(0, project_home)

from app import app as application  # Flask WSGI callable
