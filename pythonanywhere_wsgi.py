

import sys
import os
import logging

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# Project home directory on PythonAnywhere
# ============================================
project_home = '/home/mariana110195/genetic-feature-selection'

# Add project directory to Python path
if project_home not in sys.path:
    sys.path.insert(0, project_home)
    logger.info(f"Added {project_home} to Python path")

# Change working directory to project root
try:
    os.chdir(project_home)
    logger.info(f"Changed working directory to: {os.getcwd()}")
except Exception as e:
    logger.error(f"Failed to change directory: {e}")

# Create necessary directories with proper permissions
directories = [
    'uploads',
    'results',
    'results/experiments'
]

for directory in directories:
    dir_path = os.path.join(project_home, directory)
    try:
        os.makedirs(dir_path, exist_ok=True)
        os.chmod(dir_path, 0o755)
        logger.info(f"✓ Directory ready: {directory}")
    except Exception as e:
        logger.warning(f"✗ Directory creation failed for {directory}: {e}")

# Import and create Flask application
try:
    from web import create_app
    application = create_app()
    logger.info("✓ Flask application loaded successfully")
    logger.info(f"✓ Static folder: {application.static_folder}")
    logger.info(f"✓ Template folder: {application.template_folder}")
except Exception as e:
    logger.error(f"✗ Failed to load Flask application: {e}")
    raise

# Debug information
logger.info("=" * 50)
logger.info("WSGI Configuration Summary:")
logger.info(f"  - Python version: {sys.version}")
logger.info(f"  - Working directory: {os.getcwd()}")
logger.info(f"  - Project home: {project_home}")
logger.info(f"  - Python path (first 3): {sys.path[:3]}")
logger.info("=" * 50)

