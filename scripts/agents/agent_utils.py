#!/usr/bin/env python3
"""
Agent Utilities
Common utilities and helpers for agent operations
"""

import os
import sys
import json
import yaml
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import requests
from functools import wraps
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.log_step import log_step


class AgentUtils:
    """Common utilities for agents"""
    
    @staticmethod
    def get_latest_run_id(results_dir: Optional[Path] = None) -> Optional[str]:
        """Get the most recent run ID from results directory"""
        if results_dir is None:
            results_dir = Path(__file__).parent.parent.parent / "results"
        
        if not results_dir.exists():
            return None
        
        # Find directories matching run pattern
        run_dirs = [d for d in results_dir.iterdir() 
                   if d.is_dir() and d.name.startswith("run_")]
        
        if not run_dirs:
            return None
        
        # Sort by modification time
        latest = max(run_dirs, key=lambda d: d.stat().st_mtime)
        return latest.name
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    @staticmethod
    def format_file_size(bytes: int) -> str:
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes < 1024.0:
                return f"{bytes:.1f}{unit}"
            bytes /= 1024.0
        return f"{bytes:.1f}TB"
    
    @staticmethod
    def calculate_file_hash(file_path: Path, algorithm: str = "md5") -> str:
        """Calculate hash of a file"""
        hash_func = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    
    @staticmethod
    def load_json_safe(file_path: Path) -> Dict[str, Any]:
        """Safely load JSON file with error handling"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            log_step(f"Invalid JSON in {file_path}: {e}", level="error")
            return {}
        except Exception as e:
            log_step(f"Error loading {file_path}: {e}", level="error")
            return {}
    
    @staticmethod
    def load_yaml_safe(file_path: Path) -> Dict[str, Any]:
        """Safely load YAML file with error handling"""
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            log_step(f"Invalid YAML in {file_path}: {e}", level="error")
            return {}
        except Exception as e:
            log_step(f"Error loading {file_path}: {e}", level="error")
            return {}
    
    @staticmethod
    def ensure_directory(dir_path: Path) -> Path:
        """Ensure directory exists, create if necessary"""
        dir_path.mkdir(exist_ok=True, parents=True)
        return dir_path
    
    @staticmethod
    def run_command(command: List[str], timeout: int = 300) -> Tuple[bool, str, str]:
        """Run command with timeout and capture output"""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return False, "", str(e)
    
    @staticmethod
    def get_git_info() -> Dict[str, str]:
        """Get current git information"""
        info = {}
        
        # Get current commit hash
        success, stdout, _ = AgentUtils.run_command(["git", "rev-parse", "HEAD"])
        if success:
            info["commit_hash"] = stdout.strip()
        
        # Get current branch
        success, stdout, _ = AgentUtils.run_command(["git", "branch", "--show-current"])
        if success:
            info["branch"] = stdout.strip()
        
        # Check if working directory is clean
        success, stdout, _ = AgentUtils.run_command(["git", "status", "--porcelain"])
        if success:
            info["is_clean"] = len(stdout.strip()) == 0
        
        return info
    
    @staticmethod
    def send_notification(message: str, channel: str = "console", **kwargs):
        """Send notification through various channels"""
        if channel == "console":
            print(f"[NOTIFICATION] {message}")
        
        elif channel == "slack" and "webhook_url" in kwargs:
            try:
                response = requests.post(
                    kwargs["webhook_url"],
                    json={"text": message},
                    timeout=5
                )
                if response.status_code != 200:
                    log_step(f"Failed to send Slack notification: {response.status_code}", level="warning")
            except Exception as e:
                log_step(f"Error sending Slack notification: {e}", level="error")
        
        elif channel == "file" and "file_path" in kwargs:
            with open(kwargs["file_path"], 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] {message}\n")
    
    @staticmethod
    def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
        """Decorator to retry function on failure"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries - 1:
                            log_step(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...", level="warning")
                            time.sleep(current_delay)
                            current_delay *= backoff
                
                raise last_exception
            return wrapper
        return decorator
    
    @staticmethod
    def batch_process(items: List[Any], batch_size: int, process_func, **kwargs) -> List[Any]:
        """Process items in batches"""
        results = []
        total_batches = (len(items) + batch_size - 1) // batch_size
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            log_step(f"Processing batch {batch_num}/{total_batches}")
            
            try:
                batch_results = process_func(batch, **kwargs)
                results.extend(batch_results)
            except Exception as e:
                log_step(f"Error processing batch {batch_num}: {e}", level="error")
        
        return results
    
    @staticmethod
    def parallel_process(items: List[Any], process_func, max_workers: int = 4, **kwargs) -> List[Any]:
        """Process items in parallel using ThreadPoolExecutor"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(process_func, item, **kwargs): item 
                for item in items
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    log_step(f"Error processing item {item}: {e}", level="error")
                    results.append(None)
        
        return results
    
    @staticmethod
    def create_summary_table(data: List[Dict[str, Any]], columns: List[str], max_width: int = 20) -> str:
        """Create a formatted text table"""
        if not data:
            return "No data available"
        
        # Calculate column widths
        col_widths = {}
        for col in columns:
            # Header width
            col_widths[col] = len(col)
            # Data widths
            for row in data:
                value = str(row.get(col, ""))
                col_widths[col] = min(max(col_widths[col], len(value)), max_width)
        
        # Create header
        header = " | ".join(col.ljust(col_widths[col]) for col in columns)
        separator = "-+-".join("-" * col_widths[col] for col in columns)
        
        # Create rows
        rows = []
        for row_data in data:
            row = " | ".join(
                str(row_data.get(col, ""))[:col_widths[col]].ljust(col_widths[col]) 
                for col in columns
            )
            rows.append(row)
        
        # Combine
        table = [header, separator] + rows
        return "\n".join(table)
    
    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge multiple configuration dictionaries"""
        result = {}
        
        for config in configs:
            for key, value in config.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = AgentUtils.merge_configs(result[key], value)
                else:
                    result[key] = value
        
        return result


class ProgressTracker:
    """Track progress of long-running operations"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        
    def update(self, increment: int = 1):
        """Update progress"""
        self.current += increment
        self._display()
    
    def _display(self):
        """Display progress bar"""
        percent = self.current / self.total * 100
        bar_length = 40
        filled = int(bar_length * self.current / self.total)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        elapsed = time.time() - self.start_time
        if self.current > 0:
            rate = self.current / elapsed
            eta = (self.total - self.current) / rate if rate > 0 else 0
        else:
            eta = 0
        
        print(f"\r{self.description}: [{bar}] {percent:.1f}% ({self.current}/{self.total}) ETA: {AgentUtils.format_duration(eta)}", end="")
        
        if self.current >= self.total:
            print()  # New line when complete


class AgentContext:
    """Context manager for agent operations"""
    
    def __init__(self, agent_name: str, operation: str):
        self.agent_name = agent_name
        self.operation = operation
        self.start_time = None
        self.metadata = {}
        
    def __enter__(self):
        self.start_time = time.time()
        log_step(f"Starting {self.agent_name}: {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            log_step(f"Completed {self.agent_name}: {self.operation} ({AgentUtils.format_duration(duration)})")
        else:
            log_step(f"Failed {self.agent_name}: {self.operation} - {exc_val}", level="error")
        
        # Log metadata if any
        if self.metadata:
            log_step(f"Metadata: {json.dumps(self.metadata)}", level="debug")
        
        return False  # Don't suppress exceptions
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to context"""
        self.metadata[key] = value


def main():
    """Demo and utility functions"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Agent utility functions"
    )
    parser.add_argument(
        "--latest-run",
        action="store_true",
        help="Get latest run ID"
    )
    parser.add_argument(
        "--git-info",
        action="store_true",
        help="Show git information"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run utility demos"
    )
    
    args = parser.parse_args()
    
    if args.latest_run:
        run_id = AgentUtils.get_latest_run_id()
        if run_id:
            print(f"Latest run ID: {run_id}")
        else:
            print("No runs found")
    
    elif args.git_info:
        info = AgentUtils.get_git_info()
        print("Git Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    elif args.demo:
        print("Agent Utilities Demo\n")
        
        # Progress tracker demo
        print("1. Progress Tracker Demo:")
        tracker = ProgressTracker(50, "Processing items")
        for i in range(50):
            time.sleep(0.05)
            tracker.update()
        
        print("\n2. Summary Table Demo:")
        data = [
            {"agent": "test_analyzer", "status": "active", "runs": 42},
            {"agent": "workflow_optimizer", "status": "active", "runs": 108},
            {"agent": "ai_expert", "status": "inactive", "runs": 15}
        ]
        table = AgentUtils.create_summary_table(data, ["agent", "status", "runs"])
        print(table)
        
        print("\n3. Context Manager Demo:")
        with AgentContext("demo_agent", "test_operation") as ctx:
            ctx.add_metadata("test_key", "test_value")
            time.sleep(1)
            print("  Doing some work...")
        
        print("\nDemo complete!")
    
    else:
        print("Agent Utilities - Available functions:")
        print("  - get_latest_run_id()")
        print("  - format_duration()")
        print("  - format_file_size()")
        print("  - calculate_file_hash()")
        print("  - load_json_safe()")
        print("  - load_yaml_safe()")
        print("  - run_command()")
        print("  - get_git_info()")
        print("  - send_notification()")
        print("  - retry_on_failure() decorator")
        print("  - batch_process()")
        print("  - parallel_process()")
        print("  - create_summary_table()")
        print("  - ProgressTracker class")
        print("  - AgentContext manager")


if __name__ == "__main__":
    main()