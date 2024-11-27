"""
This file contains constants used throughout the project.
If you would like to use another path, just change `PROJECT_DIR`, and others will generate automatically.
"""

import os

HOME = os.path.expanduser("~")

# Change the two lines for project directory and data directory.
PROJECT_DIR = os.path.join(HOME, "workspace", "Garner")
DATA_DIR = os.path.join(HOME, "workspace", "roadnet", "data")

SG_DATA_DIR = os.path.join(DATA_DIR, "singapore")
NYC_DATA_DIR = os.path.join(DATA_DIR, "nyc")
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")
