import os
import sys

# Add the app directory to the system path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE_DIR, "app")
if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)

from ui import launch_app  # Import after updating the system path

if __name__ == "__main__":
    launch_app()