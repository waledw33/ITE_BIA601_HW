"""
SmartSelect Web Application Module
Flask-based web interface for intelligent feature optimization
Developed by: Layla Abdallah (UI/UX) & Fadi Younes (API Development)
"""

from flask import Flask

def create_app():
    """
    Application Factory Pattern
    Initializes and configures the Flask application instance
    
    Returns:
        Flask: Configured application instance
    """
    app = Flask(__name__)
    
    # Application configuration
    app.config['SECRET_KEY'] = 'smartselect-optimization-platform-2025'
    app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # Maximum 20MB file upload
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['JSON_SORT_KEYS'] = False
    
    # Register application blueprints
    from .routes import web_routes
    app.register_blueprint(web_routes)
    
    return app

__all__ = ['create_app']

