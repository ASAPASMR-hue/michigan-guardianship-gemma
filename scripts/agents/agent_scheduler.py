#!/usr/bin/env python3
"""
Agent Scheduler
Automates periodic agent runs and monitoring
"""

import os
import sys
import json
import yaml
import schedule
import time
import threading
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.log_step import log_step


@dataclass
class ScheduledJob:
    """Represents a scheduled agent job"""
    name: str
    agent_command: str
    schedule_type: str  # daily, weekly, hourly, custom
    schedule_time: str  # Time in HH:MM format or cron expression
    enabled: bool = True
    last_run: Optional[str] = None
    last_status: Optional[str] = None
    last_duration: Optional[float] = None
    run_count: int = 0
    failure_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class AgentScheduler:
    """Schedules and manages periodic agent runs"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize scheduler"""
        self.project_root = Path(__file__).parent.parent.parent
        self.scheduler_dir = self.project_root / "scheduler"
        self.scheduler_dir.mkdir(exist_ok=True, parents=True)
        
        # Configuration
        self.config_file = config_file or self.scheduler_dir / "scheduler_config.yaml"
        self.jobs_file = self.scheduler_dir / "scheduled_jobs.json"
        self.log_file = self.scheduler_dir / "scheduler.log"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("AgentScheduler")
        
        # Load configuration
        self.config = self._load_config()
        self.jobs = self._load_jobs()
        
        # Scheduler state
        self.running = False
        self.scheduler_thread = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load scheduler configuration"""
        default_config = {
            "max_concurrent_jobs": 2,
            "job_timeout": 3600,  # 1 hour
            "retry_failed_jobs": True,
            "max_retries": 3,
            "notification_email": None,
            "slack_webhook": None
        }
        
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                loaded_config = yaml.safe_load(f) or {}
                default_config.update(loaded_config)
        else:
            # Save default config
            with open(self.config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
        
        return default_config
    
    def _load_jobs(self) -> Dict[str, ScheduledJob]:
        """Load scheduled jobs"""
        jobs = {}
        
        if self.jobs_file.exists():
            with open(self.jobs_file, 'r') as f:
                job_data = json.load(f)
                for name, data in job_data.items():
                    jobs[name] = ScheduledJob(**data)
        
        return jobs
    
    def _save_jobs(self):
        """Save scheduled jobs to file"""
        job_data = {name: job.to_dict() for name, job in self.jobs.items()}
        with open(self.jobs_file, 'w') as f:
            json.dump(job_data, f, indent=2)
    
    def add_job(self, job: ScheduledJob) -> bool:
        """Add a new scheduled job"""
        if job.name in self.jobs:
            self.logger.warning(f"Job '{job.name}' already exists")
            return False
        
        self.jobs[job.name] = job
        self._save_jobs()
        self.logger.info(f"Added job: {job.name}")
        
        # Schedule if running
        if self.running:
            self._schedule_job(job)
        
        return True
    
    def remove_job(self, job_name: str) -> bool:
        """Remove a scheduled job"""
        if job_name not in self.jobs:
            self.logger.warning(f"Job '{job_name}' not found")
            return False
        
        del self.jobs[job_name]
        self._save_jobs()
        self.logger.info(f"Removed job: {job_name}")
        
        # Clear from schedule
        schedule.clear(job_name)
        
        return True
    
    def update_job(self, job_name: str, updates: Dict[str, Any]) -> bool:
        """Update an existing job"""
        if job_name not in self.jobs:
            self.logger.warning(f"Job '{job_name}' not found")
            return False
        
        job = self.jobs[job_name]
        for key, value in updates.items():
            if hasattr(job, key):
                setattr(job, key, value)
        
        self._save_jobs()
        self.logger.info(f"Updated job: {job_name}")
        
        # Reschedule if running
        if self.running:
            schedule.clear(job_name)
            self._schedule_job(job)
        
        return True
    
    def _schedule_job(self, job: ScheduledJob):
        """Schedule a job based on its configuration"""
        if not job.enabled:
            return
        
        job_runner = lambda: self._run_job(job)
        
        if job.schedule_type == "daily":
            schedule.every().day.at(job.schedule_time).do(job_runner).tag(job.name)
        elif job.schedule_type == "weekly":
            # Expect format: "monday:10:30"
            day, time = job.schedule_time.split(":")
            getattr(schedule.every(), day.lower()).at(":".join(time)).do(job_runner).tag(job.name)
        elif job.schedule_type == "hourly":
            schedule.every().hour.do(job_runner).tag(job.name)
        elif job.schedule_type == "interval":
            # Expect format: "30m" or "2h"
            interval = int(job.schedule_time[:-1])
            unit = job.schedule_time[-1]
            if unit == 'm':
                schedule.every(interval).minutes.do(job_runner).tag(job.name)
            elif unit == 'h':
                schedule.every(interval).hours.do(job_runner).tag(job.name)
        
        self.logger.info(f"Scheduled job: {job.name} ({job.schedule_type} at {job.schedule_time})")
    
    def _run_job(self, job: ScheduledJob):
        """Execute a scheduled job"""
        self.logger.info(f"Running job: {job.name}")
        start_time = time.time()
        
        try:
            # Update job status
            job.last_run = datetime.now().isoformat()
            job.run_count += 1
            
            # Prepare command
            cmd = job.agent_command.split()
            
            # Add Python interpreter if needed
            if cmd[0].endswith('.py'):
                cmd = ['python'] + cmd
            
            # Run command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config['job_timeout'],
                cwd=self.project_root
            )
            
            # Check result
            if result.returncode == 0:
                job.last_status = "success"
                self.logger.info(f"Job {job.name} completed successfully")
                
                # Save output
                output_file = self.scheduler_dir / f"{job.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                with open(output_file, 'w') as f:
                    f.write(f"=== STDOUT ===\n{result.stdout}\n")
                    f.write(f"=== STDERR ===\n{result.stderr}\n")
                
                # Send notification if configured
                self._send_notification(
                    f"Job Success: {job.name}",
                    f"Job completed in {time.time() - start_time:.1f}s"
                )
            else:
                job.last_status = "failed"
                job.failure_count += 1
                self.logger.error(f"Job {job.name} failed with code {result.returncode}")
                self.logger.error(f"Error: {result.stderr}")
                
                # Send failure notification
                self._send_notification(
                    f"Job Failed: {job.name}",
                    f"Exit code: {result.returncode}\nError: {result.stderr[:200]}"
                )
                
                # Retry if configured
                if self.config['retry_failed_jobs'] and job.failure_count <= self.config['max_retries']:
                    self.logger.info(f"Scheduling retry for {job.name} (attempt {job.failure_count})")
                    schedule.every(5).minutes.do(lambda: self._run_job(job)).tag(f"{job.name}_retry")
        
        except subprocess.TimeoutExpired:
            job.last_status = "timeout"
            job.failure_count += 1
            self.logger.error(f"Job {job.name} timed out after {self.config['job_timeout']}s")
            
            self._send_notification(
                f"Job Timeout: {job.name}",
                f"Job exceeded timeout of {self.config['job_timeout']}s"
            )
        
        except Exception as e:
            job.last_status = "error"
            job.failure_count += 1
            self.logger.error(f"Job {job.name} error: {str(e)}")
            
            self._send_notification(
                f"Job Error: {job.name}",
                f"Error: {str(e)}"
            )
        
        finally:
            job.last_duration = time.time() - start_time
            self._save_jobs()
    
    def _send_notification(self, subject: str, body: str):
        """Send notification via configured channels"""
        # Email notification
        if self.config.get('notification_email'):
            # Implement email sending
            self.logger.info(f"Would send email: {subject}")
        
        # Slack notification
        if self.config.get('slack_webhook'):
            # Implement Slack webhook
            self.logger.info(f"Would send Slack: {subject}")
    
    def start(self):
        """Start the scheduler"""
        if self.running:
            self.logger.warning("Scheduler already running")
            return
        
        self.running = True
        self.logger.info("Starting scheduler")
        
        # Schedule all enabled jobs
        for job in self.jobs.values():
            self._schedule_job(job)
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info(f"Scheduler started with {len([j for j in self.jobs.values() if j.enabled])} active jobs")
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        while self.running:
            schedule.run_pending()
            time.sleep(30)  # Check every 30 seconds
    
    def stop(self):
        """Stop the scheduler"""
        self.logger.info("Stopping scheduler")
        self.running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        schedule.clear()
        self.logger.info("Scheduler stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {
            "running": self.running,
            "total_jobs": len(self.jobs),
            "enabled_jobs": len([j for j in self.jobs.values() if j.enabled]),
            "failed_jobs": len([j for j in self.jobs.values() if j.last_status == "failed"]),
            "next_runs": self._get_next_runs(),
            "jobs": {name: job.to_dict() for name, job in self.jobs.items()}
        }
    
    def _get_next_runs(self) -> List[Dict[str, str]]:
        """Get next scheduled runs"""
        next_runs = []
        
        for job in schedule.jobs:
            if hasattr(job, 'next_run'):
                next_runs.append({
                    "job": str(job.tags),
                    "next_run": str(job.next_run)
                })
        
        return sorted(next_runs, key=lambda x: x['next_run'])
    
    def add_default_jobs(self):
        """Add default scheduled jobs"""
        default_jobs = [
            ScheduledJob(
                name="daily_quick_check",
                agent_command="scripts/agents/run_agent_pipeline.py --pipeline quick_check",
                schedule_type="daily",
                schedule_time="09:00",
                enabled=True
            ),
            ScheduledJob(
                name="weekly_system_review",
                agent_command="scripts/agents/run_agent_pipeline.py --pipeline system_review",
                schedule_type="weekly",
                schedule_time="monday:10:00",
                enabled=True
            ),
            ScheduledJob(
                name="hourly_workflow_check",
                agent_command="scripts/agents/workflow_optimizer.py --check",
                schedule_type="hourly",
                schedule_time="",
                enabled=False  # Disabled by default
            ),
            ScheduledJob(
                name="daily_dashboard_update",
                agent_command="scripts/agents/dashboard_generator.py --index",
                schedule_type="daily",
                schedule_time="18:00",
                enabled=True
            )
        ]
        
        for job in default_jobs:
            self.add_job(job)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Agent Scheduler - Automate agent runs"
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start the scheduler daemon"
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the scheduler daemon"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show scheduler status"
    )
    parser.add_argument(
        "--add-job",
        help="Add a new job (JSON format)"
    )
    parser.add_argument(
        "--remove-job",
        help="Remove a job by name"
    )
    parser.add_argument(
        "--enable-job",
        help="Enable a job by name"
    )
    parser.add_argument(
        "--disable-job",
        help="Disable a job by name"
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="List all scheduled jobs"
    )
    parser.add_argument(
        "--add-defaults",
        action="store_true",
        help="Add default scheduled jobs"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon (keeps running)"
    )
    
    args = parser.parse_args()
    
    scheduler = AgentScheduler()
    
    if args.start:
        scheduler.start()
        if args.daemon:
            print("Scheduler running as daemon. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                scheduler.stop()
        else:
            print("Scheduler started in background")
    
    elif args.stop:
        scheduler.stop()
        print("Scheduler stopped")
    
    elif args.status:
        status = scheduler.get_status()
        print(f"Scheduler Status: {'Running' if status['running'] else 'Stopped'}")
        print(f"Total Jobs: {status['total_jobs']}")
        print(f"Enabled Jobs: {status['enabled_jobs']}")
        print(f"Failed Jobs: {status['failed_jobs']}")
        
        if status['next_runs']:
            print("\nNext Scheduled Runs:")
            for run in status['next_runs'][:5]:
                print(f"  {run['job']}: {run['next_run']}")
    
    elif args.add_job:
        try:
            job_data = json.loads(args.add_job)
            job = ScheduledJob(**job_data)
            if scheduler.add_job(job):
                print(f"Added job: {job.name}")
            else:
                print("Failed to add job")
        except Exception as e:
            print(f"Error adding job: {e}")
    
    elif args.remove_job:
        if scheduler.remove_job(args.remove_job):
            print(f"Removed job: {args.remove_job}")
        else:
            print("Failed to remove job")
    
    elif args.enable_job:
        if scheduler.update_job(args.enable_job, {"enabled": True}):
            print(f"Enabled job: {args.enable_job}")
        else:
            print("Failed to enable job")
    
    elif args.disable_job:
        if scheduler.update_job(args.disable_job, {"enabled": False}):
            print(f"Disabled job: {args.disable_job}")
        else:
            print("Failed to disable job")
    
    elif args.list_jobs:
        status = scheduler.get_status()
        print("Scheduled Jobs:\n")
        for name, job in status['jobs'].items():
            status_icon = "✓" if job['enabled'] else "✗"
            print(f"{status_icon} {name}")
            print(f"   Command: {job['agent_command']}")
            print(f"   Schedule: {job['schedule_type']} at {job['schedule_time']}")
            print(f"   Last Run: {job['last_run'] or 'Never'}")
            print(f"   Status: {job['last_status'] or 'N/A'}")
            print(f"   Runs: {job['run_count']} (Failed: {job['failure_count']})")
            print()
    
    elif args.add_defaults:
        scheduler.add_default_jobs()
        print("Added default scheduled jobs")
    
    else:
        print("Agent Scheduler - Usage:")
        print("  --start         : Start the scheduler")
        print("  --stop          : Stop the scheduler")
        print("  --status        : Show scheduler status")
        print("  --list-jobs     : List all scheduled jobs")
        print("  --add-defaults  : Add default jobs")
        print("  --daemon        : Run in foreground")
        print("\nExample:")
        print('  python agent_scheduler.py --add-job \'{"name": "test", "agent_command": "workflow_optimizer.py --check", "schedule_type": "daily", "schedule_time": "10:00"}\'')


if __name__ == "__main__":
    main()