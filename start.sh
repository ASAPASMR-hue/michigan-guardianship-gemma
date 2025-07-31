#!/bin/bash
# Michigan Guardianship AI - Launch Script (Unix/Linux/macOS)

echo "===================================="
echo "Michigan Guardianship AI"
echo "===================================="
echo ""

# Function to find Python executable
find_python() {
    # Check for python3 first (preferred)
    if command -v python3 &> /dev/null; then
        echo "python3"
        return 0
    fi
    
    # Check if 'python' points to Python 3
    if command -v python &> /dev/null; then
        # Check if it's Python 3.x
        if python --version 2>&1 | grep -q "Python 3"; then
            echo "python"
            return 0
        fi
    fi
    
    # Check common Python 3 aliases
    for py in python3.12 python3.11 python3.10 python3.9 python3.8; do
        if command -v $py &> /dev/null; then
            echo "$py"
            return 0
        fi
    done
    
    return 1
}

# Find Python executable
PYTHON_CMD=$(find_python)

if [ $? -ne 0 ]; then
    echo "Error: Python 3.8 or higher is required but not found."
    echo "Please install Python from https://www.python.org/"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if ! $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "Error: Python $REQUIRED_VERSION or higher is required. You have Python $PYTHON_VERSION"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo ""
    echo "No .env file found."
    echo "Please run '$PYTHON_CMD setup.py' first to configure your environment."
    exit 1
fi

# Check if dependencies are installed
if ! $PYTHON_CMD -c "import flask" &> /dev/null; then
    echo ""
    echo "Dependencies not installed."
    echo "Please run '$PYTHON_CMD setup.py' first to install dependencies."
    exit 1
fi

echo ""
echo "Starting the Michigan Guardianship AI server..."
echo ""
echo "The application will open at: http://127.0.0.1:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Optional: Try to open browser automatically
if command -v open &> /dev/null; then
    # macOS
    (sleep 2 && open http://127.0.0.1:5000) &
elif command -v xdg-open &> /dev/null; then
    # Linux with desktop environment
    (sleep 2 && xdg-open http://127.0.0.1:5000) &
fi

# Start the Flask application
$PYTHON_CMD app.py