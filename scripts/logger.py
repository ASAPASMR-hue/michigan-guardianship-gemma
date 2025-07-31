#!/usr/bin/env python3
"""
Simple logging utility for Phase 3 testing
"""
import datetime
from pathlib import Path

def log_step(message: str, level: str = "info"):
    """Simple logging function that prints and saves to file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format the message
    if level.lower() == "error":
        prefix = "❌ ERROR"
    elif level.lower() == "warning":
        prefix = "⚠️  WARNING"
    else:
        prefix = "ℹ️  INFO"
    
    log_message = f"[{timestamp}] {prefix}: {message}"
    
    # Print to console
    print(log_message)
    
    # Save to log file
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"phase3_testing_{datetime.date.today()}.log"
    with open(log_file, "a") as f:
        f.write(log_message + "\n")