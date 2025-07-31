#!/usr/bin/env python3
"""
Michigan Guardianship AI - Web Server
Flask application serving the chat interface
"""

import os
import sys
import json
import logging
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from flask_session import Session
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from scripts.production_pipeline import GuardianshipRAG
from server.conversation_state import ConversationState
from server.state_extractor import StateExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='ui')
CORS(app, origins=['http://localhost:5000', 'http://127.0.0.1:5000'], supports_credentials=True)

# Configure session
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session/'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
app.config['SESSION_COOKIE_SECURE'] = False  # Set True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Initialize Flask-Session
Session(app)

# Create session directory if it doesn't exist
os.makedirs('./flask_session/', exist_ok=True)

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
    """Main API endpoint for answering questions - returns structured response with session state"""
    try:
        # Initialize RAG if needed
        initialize_rag()
        
        # Get or create session ID
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
            session['conversation_state'] = ConversationState().to_json()
            logger.info(f"New session created: {session['session_id']}")
        
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
        
        # Log the question with session ID
        logger.info(f"Session {session['session_id'][:8]}: {question[:100]}...")
        
        # Load current conversation state
        current_state = ConversationState.from_json(session.get('conversation_state', '{}'))
        
        # Get structured answer from RAG pipeline with conversation state
        result = rag_pipeline.get_answer(question, conversation_state=current_state)
        
        # Extract state from the exchange
        if 'data' in result and 'answer_markdown' in result['data']:
            # Extract facts from this exchange
            new_state = StateExtractor.extract_from_exchange(
                user_question=question,
                assistant_response=result['data']['answer_markdown'],
                current_state=current_state
            )
            
            # Update session state
            session['conversation_state'] = new_state.to_json()
            
            # Add state info to response
            result['data']['conversation_state'] = new_state.model_dump()
            result['data']['state_updates'] = {
                'extracted_facts': new_state.get_summary(),
                'has_context': new_state.has_meaningful_context()
            }
            
            # Backward compatibility
            result['answer'] = result['data']['answer_markdown']
        
        # Include session ID in response
        result['session_id'] = session['session_id']
        
        logger.info(f"Session {session['session_id'][:8]}: Response generated with state")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/ask/structured', methods=['POST'])
def ask_question_structured():
    """API endpoint that returns only structured data format"""
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
        logger.info(f"Received structured request: {question[:100]}...")
        
        # Get structured answer from RAG pipeline
        result = rag_pipeline.get_answer(question)
        
        # Return only the structured data portion
        if 'data' in result:
            return jsonify({
                "data": result['data'],
                "metadata": result.get('metadata', {}),
                "timestamp": result.get('timestamp', datetime.now().isoformat())
            }), 200
        else:
            # Fallback if structure is not available
            return jsonify({
                "error": "Structured response not available",
                "details": "The system returned an unstructured response"
            }), 500
        
    except Exception as e:
        logger.error(f"Error processing structured question: {e}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/session/state', methods=['GET'])
def get_session_state():
    """Get current conversation state"""
    try:
        if 'session_id' not in session:
            return jsonify({
                "session_id": None,
                "conversation_state": None,
                "message": "No active session"
            }), 200
        
        current_state = ConversationState.from_json(session.get('conversation_state', '{}'))
        
        return jsonify({
            "session_id": session['session_id'],
            "conversation_state": current_state.model_dump(),
            "context_string": current_state.to_context_string() if current_state.has_meaningful_context() else None,
            "summary": current_state.get_summary()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting session state: {e}")
        return jsonify({
            "error": "Failed to get session state",
            "details": str(e)
        }), 500

@app.route('/api/session/clear', methods=['POST'])
def clear_session():
    """Clear the current session state"""
    try:
        if 'session_id' in session:
            old_session_id = session['session_id']
            session.clear()
            logger.info(f"Session cleared: {old_session_id}")
            
            return jsonify({
                "status": "success",
                "message": "Session cleared successfully",
                "old_session_id": old_session_id
            }), 200
        else:
            return jsonify({
                "status": "success",
                "message": "No active session to clear"
            }), 200
            
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        return jsonify({
            "error": "Failed to clear session",
            "details": str(e)
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