"""
SmartSelect Platform - Application Bootstrap
Intelligent Feature Optimization System

Launch command: python app.py
Production deployment: gunicorn -w 4 -b 0.0.0.0:5000 app:app
"""

import os
from web import create_app

# Initialize Flask application instance
app = create_app()

if __name__ == '__main__':
    # Ensure required directory structure exists
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('results/experiments', exist_ok=True)
    
    # Launch development server
    # Note: For production environments, use a WSGI server like Gunicorn
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,  # Disable in production for security
        threaded=True
    )

