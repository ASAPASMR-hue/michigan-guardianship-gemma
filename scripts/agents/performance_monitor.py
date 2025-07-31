#!/usr/bin/env python3
"""
Agent Performance Monitor
Tracks and analyzes agent execution performance over time
"""

import os
import sys
import json
import yaml
import time
import psutil
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import statistics

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.log_step import log_step


@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    timestamp: str
    agent_name: str
    operation: str
    duration: float
    memory_used: float
    cpu_percent: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentProfile:
    """Performance profile for an agent"""
    name: str
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_duration: float = 0.0
    avg_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    avg_memory: float = 0.0
    avg_cpu: float = 0.0
    error_types: Dict[str, int] = field(default_factory=dict)
    recent_metrics: List[PerformanceMetric] = field(default_factory=list)
    
    def update(self, metric: PerformanceMetric):
        """Update profile with new metric"""
        self.total_runs += 1
        
        if metric.success:
            self.successful_runs += 1
        else:
            self.failed_runs += 1
            if metric.error:
                self.error_types[metric.error] = self.error_types.get(metric.error, 0) + 1
        
        self.total_duration += metric.duration
        self.avg_duration = self.total_duration / self.total_runs
        self.min_duration = min(self.min_duration, metric.duration)
        self.max_duration = max(self.max_duration, metric.duration)
        
        # Keep recent metrics (last 100)
        self.recent_metrics.append(metric)
        if len(self.recent_metrics) > 100:
            self.recent_metrics.pop(0)
        
        # Update averages from recent metrics
        if self.recent_metrics:
            self.avg_memory = statistics.mean(m.memory_used for m in self.recent_metrics)
            self.avg_cpu = statistics.mean(m.cpu_percent for m in self.recent_metrics)


class PerformanceMonitor:
    """Monitors and analyzes agent performance"""
    
    def __init__(self):
        """Initialize performance monitor"""
        self.project_root = Path(__file__).parent.parent.parent
        self.monitor_dir = self.project_root / "monitoring"
        self.monitor_dir.mkdir(exist_ok=True, parents=True)
        
        # Storage
        self.metrics_file = self.monitor_dir / "performance_metrics.jsonl"
        self.profiles_file = self.monitor_dir / "agent_profiles.json"
        self.alerts_file = self.monitor_dir / "performance_alerts.json"
        
        # State
        self.agent_profiles = self._load_profiles()
        self.active_monitors = {}
        self.alert_thresholds = self._load_alert_thresholds()
        self.recent_alerts = deque(maxlen=100)
        
        # System monitoring
        self.system_metrics = {
            "cpu_history": deque(maxlen=60),  # Last 60 measurements
            "memory_history": deque(maxlen=60),
            "disk_history": deque(maxlen=60)
        }
    
    def _load_profiles(self) -> Dict[str, AgentProfile]:
        """Load agent profiles from storage"""
        profiles = {}
        
        if self.profiles_file.exists():
            with open(self.profiles_file, 'r') as f:
                profile_data = json.load(f)
                for name, data in profile_data.items():
                    # Convert back to AgentProfile
                    recent_metrics = [PerformanceMetric(**m) for m in data.pop('recent_metrics', [])]
                    profile = AgentProfile(**data)
                    profile.recent_metrics = recent_metrics
                    profiles[name] = profile
        
        return profiles
    
    def _save_profiles(self):
        """Save agent profiles to storage"""
        profile_data = {}
        for name, profile in self.agent_profiles.items():
            data = asdict(profile)
            # Convert metrics to dict
            data['recent_metrics'] = [asdict(m) for m in profile.recent_metrics]
            profile_data[name] = data
        
        with open(self.profiles_file, 'w') as f:
            json.dump(profile_data, f, indent=2)
    
    def _load_alert_thresholds(self) -> Dict[str, Any]:
        """Load alert thresholds"""
        default_thresholds = {
            "duration_multiplier": 2.0,  # Alert if duration > 2x average
            "memory_limit_mb": 1000,     # Alert if memory > 1GB
            "cpu_percent": 80,           # Alert if CPU > 80%
            "failure_rate": 0.2,         # Alert if failure rate > 20%
            "consecutive_failures": 3     # Alert after 3 consecutive failures
        }
        
        thresholds_file = self.monitor_dir / "alert_thresholds.yaml"
        if thresholds_file.exists():
            with open(thresholds_file, 'r') as f:
                loaded = yaml.safe_load(f) or {}
                default_thresholds.update(loaded)
        else:
            # Save defaults
            with open(thresholds_file, 'w') as f:
                yaml.dump(default_thresholds, f, default_flow_style=False)
        
        return default_thresholds
    
    def start_monitoring(self, agent_name: str, operation: str) -> str:
        """Start monitoring an agent operation"""
        monitor_id = f"{agent_name}_{operation}_{datetime.now().timestamp()}"
        
        # Get initial system state
        process = psutil.Process()
        
        self.active_monitors[monitor_id] = {
            "agent_name": agent_name,
            "operation": operation,
            "start_time": time.time(),
            "start_memory": process.memory_info().rss / 1024 / 1024,  # MB
            "start_cpu": process.cpu_percent(interval=0.1)
        }
        
        log_step(f"Started monitoring: {agent_name} - {operation}")
        return monitor_id
    
    def stop_monitoring(self, monitor_id: str, success: bool = True, error: Optional[str] = None, metadata: Dict[str, Any] = None):
        """Stop monitoring and record metrics"""
        if monitor_id not in self.active_monitors:
            log_step(f"Monitor ID not found: {monitor_id}", level="warning")
            return
        
        monitor = self.active_monitors.pop(monitor_id)
        end_time = time.time()
        
        # Get final system state
        process = psutil.Process()
        end_memory = process.memory_info().rss / 1024 / 1024
        end_cpu = process.cpu_percent(interval=0.1)
        
        # Create metric
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            agent_name=monitor["agent_name"],
            operation=monitor["operation"],
            duration=end_time - monitor["start_time"],
            memory_used=end_memory - monitor["start_memory"],
            cpu_percent=(monitor["start_cpu"] + end_cpu) / 2,
            success=success,
            error=error,
            metadata=metadata or {}
        )
        
        # Save metric
        self._save_metric(metric)
        
        # Update profile
        self._update_profile(metric)
        
        # Check for alerts
        self._check_alerts(metric)
        
        log_step(f"Stopped monitoring: {monitor['agent_name']} - {monitor['operation']} (Duration: {metric.duration:.2f}s)")
    
    def _save_metric(self, metric: PerformanceMetric):
        """Save metric to storage"""
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(asdict(metric)) + '\n')
    
    def _update_profile(self, metric: PerformanceMetric):
        """Update agent profile with new metric"""
        if metric.agent_name not in self.agent_profiles:
            self.agent_profiles[metric.agent_name] = AgentProfile(name=metric.agent_name)
        
        self.agent_profiles[metric.agent_name].update(metric)
        self._save_profiles()
    
    def _check_alerts(self, metric: PerformanceMetric):
        """Check if metric triggers any alerts"""
        alerts = []
        profile = self.agent_profiles.get(metric.agent_name)
        
        if not profile or profile.total_runs < 5:
            # Not enough data for meaningful alerts
            return
        
        # Duration alert
        if metric.duration > profile.avg_duration * self.alert_thresholds["duration_multiplier"]:
            alerts.append({
                "type": "slow_execution",
                "severity": "warning",
                "message": f"{metric.agent_name} took {metric.duration:.1f}s (avg: {profile.avg_duration:.1f}s)"
            })
        
        # Memory alert
        if metric.memory_used > self.alert_thresholds["memory_limit_mb"]:
            alerts.append({
                "type": "high_memory",
                "severity": "warning",
                "message": f"{metric.agent_name} used {metric.memory_used:.1f}MB memory"
            })
        
        # CPU alert
        if metric.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append({
                "type": "high_cpu",
                "severity": "warning",
                "message": f"{metric.agent_name} used {metric.cpu_percent:.1f}% CPU"
            })
        
        # Failure rate alert
        failure_rate = profile.failed_runs / profile.total_runs
        if failure_rate > self.alert_thresholds["failure_rate"]:
            alerts.append({
                "type": "high_failure_rate",
                "severity": "error",
                "message": f"{metric.agent_name} has {failure_rate:.1%} failure rate"
            })
        
        # Consecutive failures
        recent_failures = sum(1 for m in profile.recent_metrics[-3:] if not m.success)
        if recent_failures >= self.alert_thresholds["consecutive_failures"]:
            alerts.append({
                "type": "consecutive_failures",
                "severity": "critical",
                "message": f"{metric.agent_name} failed {recent_failures} times in a row"
            })
        
        # Process alerts
        for alert in alerts:
            alert["timestamp"] = datetime.now().isoformat()
            alert["agent_name"] = metric.agent_name
            alert["metric_id"] = f"{metric.timestamp}_{metric.agent_name}_{metric.operation}"
            
            self.recent_alerts.append(alert)
            log_step(f"ALERT [{alert['severity']}]: {alert['message']}", level="warning")
        
        # Save alerts
        if alerts:
            with open(self.alerts_file, 'a') as f:
                for alert in alerts:
                    f.write(json.dumps(alert) + '\n')
    
    def get_agent_stats(self, agent_name: str) -> Dict[str, Any]:
        """Get performance statistics for an agent"""
        if agent_name not in self.agent_profiles:
            return {"error": f"No data for agent: {agent_name}"}
        
        profile = self.agent_profiles[agent_name]
        
        return {
            "name": agent_name,
            "total_runs": profile.total_runs,
            "success_rate": profile.successful_runs / profile.total_runs if profile.total_runs > 0 else 0,
            "performance": {
                "avg_duration": profile.avg_duration,
                "min_duration": profile.min_duration,
                "max_duration": profile.max_duration,
                "avg_memory_mb": profile.avg_memory,
                "avg_cpu_percent": profile.avg_cpu
            },
            "errors": dict(profile.error_types),
            "recent_trend": self._calculate_trend(profile.recent_metrics)
        }
    
    def _calculate_trend(self, metrics: List[PerformanceMetric]) -> Dict[str, str]:
        """Calculate performance trend from recent metrics"""
        if len(metrics) < 10:
            return {"status": "insufficient_data"}
        
        # Compare last 5 to previous 5
        recent = metrics[-5:]
        previous = metrics[-10:-5]
        
        recent_avg = statistics.mean(m.duration for m in recent)
        previous_avg = statistics.mean(m.duration for m in previous)
        
        if recent_avg < previous_avg * 0.9:
            trend = "improving"
        elif recent_avg > previous_avg * 1.1:
            trend = "degrading"
        else:
            trend = "stable"
        
        return {
            "status": trend,
            "recent_avg": recent_avg,
            "previous_avg": previous_avg,
            "change_percent": ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        # Update system metrics
        self.system_metrics["cpu_history"].append(psutil.cpu_percent(interval=0.1))
        self.system_metrics["memory_history"].append(psutil.virtual_memory().percent)
        self.system_metrics["disk_history"].append(psutil.disk_usage('/').percent)
        
        # Calculate agent health
        agent_health = {}
        for name, profile in self.agent_profiles.items():
            if profile.total_runs > 0:
                agent_health[name] = {
                    "success_rate": profile.successful_runs / profile.total_runs,
                    "avg_duration": profile.avg_duration,
                    "recent_failures": sum(1 for m in profile.recent_metrics[-10:] if not m.success)
                }
        
        # Overall health score (0-100)
        health_score = 100
        
        # Deduct for high failure rates
        avg_success_rate = statistics.mean(a["success_rate"] for a in agent_health.values()) if agent_health else 1.0
        health_score -= (1 - avg_success_rate) * 30
        
        # Deduct for system resource usage
        if self.system_metrics["cpu_history"]:
            avg_cpu = statistics.mean(self.system_metrics["cpu_history"])
            if avg_cpu > 80:
                health_score -= 10
        
        if self.system_metrics["memory_history"]:
            avg_memory = statistics.mean(self.system_metrics["memory_history"])
            if avg_memory > 80:
                health_score -= 10
        
        # Deduct for recent alerts
        critical_alerts = sum(1 for a in self.recent_alerts if a["severity"] == "critical")
        health_score -= critical_alerts * 5
        
        return {
            "health_score": max(0, health_score),
            "system_metrics": {
                "cpu_percent": self.system_metrics["cpu_history"][-1] if self.system_metrics["cpu_history"] else 0,
                "memory_percent": self.system_metrics["memory_history"][-1] if self.system_metrics["memory_history"] else 0,
                "disk_percent": self.system_metrics["disk_history"][-1] if self.system_metrics["disk_history"] else 0
            },
            "agent_health": agent_health,
            "recent_alerts": list(self.recent_alerts)[-10:],  # Last 10 alerts
            "active_monitors": len(self.active_monitors)
        }
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        report = f"""# Agent Performance Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## System Health

"""
        
        health = self.get_system_health()
        
        # Health score with emoji
        score = health["health_score"]
        if score >= 90:
            emoji = "🟢"
        elif score >= 70:
            emoji = "🟡"
        else:
            emoji = "🔴"
        
        report += f"**Overall Health Score**: {emoji} {score}/100\n\n"
        
        # System metrics
        report += "### System Resources\n"
        report += f"- CPU Usage: {health['system_metrics']['cpu_percent']:.1f}%\n"
        report += f"- Memory Usage: {health['system_metrics']['memory_percent']:.1f}%\n"
        report += f"- Disk Usage: {health['system_metrics']['disk_percent']:.1f}%\n\n"
        
        # Agent performance
        report += "## Agent Performance\n\n"
        report += "| Agent | Runs | Success Rate | Avg Duration | Trend |\n"
        report += "|-------|------|--------------|--------------|-------|\n"
        
        for name, profile in sorted(self.agent_profiles.items(), key=lambda x: x[1].total_runs, reverse=True):
            if profile.total_runs > 0:
                stats = self.get_agent_stats(name)
                success_rate = stats["success_rate"] * 100
                trend = stats["recent_trend"]["status"]
                
                # Trend emoji
                if trend == "improving":
                    trend_emoji = "📈"
                elif trend == "degrading":
                    trend_emoji = "📉"
                else:
                    trend_emoji = "➡️"
                
                report += f"| {name[:20]} | {profile.total_runs} | {success_rate:.1f}% | "
                report += f"{profile.avg_duration:.2f}s | {trend_emoji} {trend} |\n"
        
        # Recent alerts
        if self.recent_alerts:
            report += "\n## Recent Alerts\n\n"
            
            # Group by severity
            by_severity = defaultdict(list)
            for alert in list(self.recent_alerts)[-20:]:  # Last 20
                by_severity[alert["severity"]].append(alert)
            
            for severity in ["critical", "error", "warning"]:
                if severity in by_severity:
                    report += f"### {severity.upper()}\n"
                    for alert in by_severity[severity][-5:]:  # Last 5 per severity
                        report += f"- [{alert['timestamp'][:19]}] {alert['message']}\n"
                    report += "\n"
        
        # Recommendations
        report += "## Recommendations\n\n"
        
        recommendations = []
        
        # Check for poor performers
        for name, profile in self.agent_profiles.items():
            if profile.total_runs >= 10:
                failure_rate = profile.failed_runs / profile.total_runs
                if failure_rate > 0.2:
                    recommendations.append(f"- **{name}**: High failure rate ({failure_rate:.1%}). Review error logs and implementation.")
                
                if profile.avg_duration > 30:
                    recommendations.append(f"- **{name}**: Slow execution (avg {profile.avg_duration:.1f}s). Consider optimization.")
        
        # System recommendations
        if health["system_metrics"]["memory_percent"] > 80:
            recommendations.append("- **System**: High memory usage. Consider increasing resources or optimizing agents.")
        
        if not recommendations:
            recommendations.append("- All agents performing within acceptable parameters ✅")
        
        for rec in recommendations:
            report += rec + "\n"
        
        report += "\n## Top Error Types\n\n"
        
        # Aggregate all errors
        all_errors = defaultdict(int)
        for profile in self.agent_profiles.values():
            for error, count in profile.error_types.items():
                all_errors[error] += count
        
        if all_errors:
            sorted_errors = sorted(all_errors.items(), key=lambda x: x[1], reverse=True)
            for error, count in sorted_errors[:5]:
                report += f"- {error}: {count} occurrences\n"
        else:
            report += "No errors recorded ✅\n"
        
        return report
    
    def cleanup_old_metrics(self, days: int = 30):
        """Clean up metrics older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Read all metrics
        kept_metrics = []
        removed_count = 0
        
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                for line in f:
                    metric = json.loads(line)
                    metric_date = datetime.fromisoformat(metric["timestamp"])
                    
                    if metric_date > cutoff_date:
                        kept_metrics.append(line.strip())
                    else:
                        removed_count += 1
        
        # Rewrite file with kept metrics
        with open(self.metrics_file, 'w') as f:
            for line in kept_metrics:
                f.write(line + '\n')
        
        log_step(f"Cleaned up {removed_count} metrics older than {days} days")


# Context manager for easy monitoring
class monitor_agent:
    """Context manager for monitoring agent operations"""
    
    def __init__(self, agent_name: str, operation: str, monitor: Optional[PerformanceMonitor] = None):
        self.agent_name = agent_name
        self.operation = operation
        self.monitor = monitor or PerformanceMonitor()
        self.monitor_id = None
        self.success = True
        self.error = None
        self.metadata = {}
    
    def __enter__(self):
        self.monitor_id = self.monitor.start_monitoring(self.agent_name, self.operation)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.success = False
            self.error = str(exc_val)
        
        self.monitor.stop_monitoring(
            self.monitor_id,
            success=self.success,
            error=self.error,
            metadata=self.metadata
        )
        
        # Don't suppress exceptions
        return False
    
    def set_metadata(self, key: str, value: Any):
        """Add metadata to the monitoring session"""
        self.metadata[key] = value


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Agent Performance Monitor"
    )
    parser.add_argument(
        "--stats",
        help="Show stats for specific agent"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Show system health"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate performance report"
    )
    parser.add_argument(
        "--cleanup",
        type=int,
        help="Clean up metrics older than N days"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run monitoring demo"
    )
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor()
    
    if args.stats:
        stats = monitor.get_agent_stats(args.stats)
        print(json.dumps(stats, indent=2))
    
    elif args.health:
        health = monitor.get_system_health()
        print(f"System Health Score: {health['health_score']}/100")
        print(f"Active Monitors: {health['active_monitors']}")
        print(f"Recent Alerts: {len(health['recent_alerts'])}")
        
        print("\nSystem Metrics:")
        for key, value in health['system_metrics'].items():
            print(f"  {key}: {value:.1f}%")
    
    elif args.report:
        report = monitor.generate_performance_report()
        print(report)
        
        # Save report
        report_path = monitor.monitor_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
    
    elif args.cleanup:
        monitor.cleanup_old_metrics(args.cleanup)
    
    elif args.demo:
        print("Running monitoring demo...")
        
        # Simulate agent operations
        agents = ["test_analyzer", "workflow_optimizer", "ai_expert"]
        
        for agent in agents:
            for i in range(3):
                # Use context manager
                with monitor_agent(agent, f"operation_{i}", monitor) as m:
                    # Simulate work
                    time.sleep(0.5 + i * 0.2)
                    m.set_metadata("iteration", i)
                    
                    # Simulate occasional failure
                    if i == 2 and agent == "ai_expert":
                        raise Exception("Simulated error")
        
        print("\nDemo complete. Generating report...")
        report = monitor.generate_performance_report()
        print(report)
    
    else:
        print("Performance Monitor - Usage:")
        print("  --stats AGENT    : Show stats for agent")
        print("  --health         : Show system health")
        print("  --report         : Generate report")
        print("  --cleanup DAYS   : Clean old metrics")
        print("  --demo           : Run monitoring demo")


if __name__ == "__main__":
    main()