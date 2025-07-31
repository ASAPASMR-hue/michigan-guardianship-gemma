#!/usr/bin/env python3
"""
Michigan Guardianship AI - Web Server
Flask application serving the chat interface
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from scripts.production_pipeline import GuardianshipRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='ui')
CORS(app, origins=['http://localhost:5000', 'http://127.0.0.1:5000'])

# Initialize the RAG pipeline
rag_pipeline = None

def initialize_rag():
    """Initialize the RAG pipeline on first request"""
    global rag_pipeline
    if rag_pipeline is None:
        logger.info("Initializing RAG pipeline...")
        try:
            # Use Gemma 3 4B IT model - multimodal, 128K context, instruction-tuned
            rag_pipeline = GuardianshipRAG(model_name="google/gemma-3-4b-it")
            logger.info("RAG pipeline initialized with Gemma 3 4B IT model")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            # Fall back to smaller model if needed
            try:
                logger.info("Attempting fallback to Gemma 3 1B IT model...")
                rag_pipeline = GuardianshipRAG(model_name="google/gemma-3-1b-it")
                logger.info("RAG pipeline initialized with Gemma 3 1B IT model")
            except Exception as fallback_e:
                logger.error(f"Fallback initialization also failed: {fallback_e}")
                raise

@app.route('/')
def index():
    """Serve the main chat interface"""
    return send_from_directory('ui', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from ui directory"""
    return send_from_directory('ui', path)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        initialize_rag()
        health_status = rag_pipeline.health_check()
        return jsonify(health_status), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 503

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Main API endpoint for answering questions"""
    try:
        # Initialize RAG if needed
        initialize_rag()
        
        # Get question from request
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                "error": "No question provided",
                "details": "Please include a 'question' field in your request"
            }), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({
                "error": "Empty question",
                "details": "Please provide a valid question"
            }), 400
        
        # Log the question
        logger.info(f"Received question: {question[:100]}...")
        
        # Get answer from RAG pipeline
        result = rag_pipeline.get_answer(question)
        
        # Format response
        response = {
            "answer": result.get("answer", "I'm sorry, I couldn't process your question."),
            "metadata": result.get("metadata", {}),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Successfully generated response")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Optional endpoint for collecting user feedback"""
    try:
        data = request.get_json()
        
        # Log feedback (in production, save to database)
        logger.info(f"Feedback received: {data}")
        
        return jsonify({
            "status": "success",
            "message": "Thank you for your feedback!"
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return jsonify({
            "error": "Failed to submit feedback",
            "details": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Not found",
        "message": "The requested resource was not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500

def main():
    """Main entry point"""
    # Check for required environment variables
    required_vars = ['GOOGLE_API_KEY', 'HUGGINGFACE_TOKEN']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"\n⚠️  Missing required environment variables: {', '.join(missing_vars)}")
        print("Please run 'python setup.py' to configure your environment.\n")
        sys.exit(1)
    
    # Start the server
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '127.0.0.1')
    
    print("\n" + "="*60)
    print("Michigan Guardianship AI - Web Server")
    print("="*60)
    print(f"\n✅ Server starting on http://{host}:{port}")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Run the Flask app
    app.run(
        host=host,
        port=port,
        debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true',
        threaded=True
    )

if __name__ == '__main__':
    main()