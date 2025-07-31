#!/usr/bin/env python3
"""
Agent Configuration Manager
Centralized configuration management for all agents
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import copy
from dataclasses import dataclass, field, asdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.log_step import log_step


@dataclass
class AgentConfig:
    """Configuration for a single agent"""
    name: str
    enabled: bool = True
    version: str = "1.0.0"
    parameters: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """Create from dictionary"""
        return cls(**data)


class ConfigManager:
    """Manages configuration for all agents"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration manager"""
        self.project_root = Path(__file__).parent.parent.parent
        self.config_dir = config_dir or self.project_root / "agent_configs"
        self.config_dir.mkdir(exist_ok=True, parents=True)
        
        # Configuration files
        self.global_config_file = self.config_dir / "global_config.yaml"
        self.agents_config_file = self.config_dir / "agents_config.json"
        self.profiles_dir = self.config_dir / "profiles"
        self.profiles_dir.mkdir(exist_ok=True)
        
        # Load configurations
        self.global_config = self._load_global_config()
        self.agent_configs = self._load_agent_configs()
        self.config_profiles = self._load_profiles()
        
        # Configuration history
        self.history_file = self.config_dir / "config_history.jsonl"
        
        # Initialize default configurations if needed
        self._initialize_defaults()
    
    def _load_global_config(self) -> Dict[str, Any]:
        """Load global configuration"""
        default_global = {
            "project_name": "Michigan Guardianship AI",
            "environment": "development",
            "log_level": "INFO",
            "max_concurrent_agents": 5,
            "default_timeout": 300,
            "monitoring": {
                "enabled": True,
                "performance_tracking": True,
                "alert_on_failure": True
            },
            "security": {
                "api_key_encryption": False,
                "secure_storage": False
            },
            "features": {
                "dashboard": True,
                "scheduler": True,
                "performance_monitor": True
            }
        }
        
        if self.global_config_file.exists():
            with open(self.global_config_file, 'r') as f:
                loaded = yaml.safe_load(f) or {}
                default_global.update(loaded)
        else:
            # Save defaults
            with open(self.global_config_file, 'w') as f:
                yaml.dump(default_global, f, default_flow_style=False)
        
        return default_global
    
    def _load_agent_configs(self) -> Dict[str, AgentConfig]:
        """Load agent configurations"""
        configs = {}
        
        if self.agents_config_file.exists():
            with open(self.agents_config_file, 'r') as f:
                data = json.load(f)
                for name, config_data in data.items():
                    configs[name] = AgentConfig.from_dict(config_data)
        
        return configs
    
    def _save_agent_configs(self):
        """Save agent configurations"""
        data = {name: config.to_dict() for name, config in self.agent_configs.items()}
        with open(self.agents_config_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load configuration profiles"""
        profiles = {}
        
        # Load built-in profiles
        builtin_profiles = {
            "development": {
                "global": {
                    "environment": "development",
                    "log_level": "DEBUG",
                    "monitoring.performance_tracking": True
                },
                "agents": {
                    "*": {
                        "parameters.debug": True,
                        "resource_limits.timeout": 600
                    }
                }
            },
            "production": {
                "global": {
                    "environment": "production",
                    "log_level": "WARNING",
                    "monitoring.alert_on_failure": True,
                    "security.api_key_encryption": True
                },
                "agents": {
                    "*": {
                        "parameters.debug": False,
                        "resource_limits.timeout": 300,
                        "resource_limits.memory_mb": 1024
                    }
                }
            },
            "testing": {
                "global": {
                    "environment": "testing",
                    "log_level": "INFO",
                    "max_concurrent_agents": 2
                },
                "agents": {
                    "*": {
                        "parameters.test_mode": True,
                        "resource_limits.timeout": 60
                    }
                }
            }
        }
        
        profiles.update(builtin_profiles)
        
        # Load custom profiles
        for profile_file in self.profiles_dir.glob("*.yaml"):
            with open(profile_file, 'r') as f:
                profile_data = yaml.safe_load(f) or {}
                profiles[profile_file.stem] = profile_data
        
        return profiles
    
    def _initialize_defaults(self):
        """Initialize default agent configurations"""
        default_agents = {
            "test_analyzer": AgentConfig(
                name="test_analyzer",
                parameters={
                    "max_runs_to_compare": 5,
                    "generate_visualizations": True
                },
                resource_limits={"timeout": 300, "memory_mb": 512}
            ),
            "workflow_optimizer": AgentConfig(
                name="workflow_optimizer",
                parameters={
                    "golden_question_count": 5,
                    "check_interval": 300
                },
                features={"auto_suggest": True}
            ),
            "ai_expert": AgentConfig(
                name="ai_expert",
                parameters={
                    "optimization_iterations": 3,
                    "test_variations": 5
                },
                dependencies=["adaptive_retrieval", "llm_handler"]
            ),
            "code_refactorer": AgentConfig(
                name="code_refactorer",
                parameters={
                    "complexity_threshold": 10,
                    "min_docstring_coverage": 0.8
                }
            ),
            "system_architect": AgentConfig(
                name="system_architect",
                parameters={
                    "generate_diagrams": True,
                    "analyze_dependencies": True
                },
                dependencies=["graphviz"]
            ),
            "product_strategist": AgentConfig(
                name="product_strategist",
                parameters={
                    "analyze_competitors": True,
                    "feature_scoring_method": "weighted"
                }
            ),
            "analytics_engineer": AgentConfig(
                name="analytics_engineer",
                parameters={
                    "visualization_backend": "plotly",
                    "correlation_threshold": 0.5
                },
                dependencies=["matplotlib", "seaborn", "plotly"]
            ),
            "dashboard_generator": AgentConfig(
                name="dashboard_generator",
                parameters={
                    "template_engine": "jinja2",
                    "chart_library": "plotly"
                }
            ),
            "agent_scheduler": AgentConfig(
                name="agent_scheduler",
                parameters={
                    "max_concurrent_jobs": 3,
                    "job_timeout": 3600
                }
            ),
            "performance_monitor": AgentConfig(
                name="performance_monitor",
                parameters={
                    "metric_retention_days": 30,
                    "alert_threshold_multiplier": 2.0
                }
            )
        }
        
        # Add defaults if not exists
        for name, config in default_agents.items():
            if name not in self.agent_configs:
                self.agent_configs[name] = config
        
        self._save_agent_configs()
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent"""
        return self.agent_configs.get(agent_name)
    
    def update_agent_config(self, agent_name: str, updates: Dict[str, Any], record_history: bool = True) -> bool:
        """Update agent configuration"""
        if agent_name not in self.agent_configs:
            log_step(f"Agent '{agent_name}' not found", level="warning")
            return False
        
        # Record history
        if record_history:
            self._record_config_change(
                agent_name,
                "before",
                self.agent_configs[agent_name].to_dict()
            )
        
        # Apply updates
        config = self.agent_configs[agent_name]
        for key, value in updates.items():
            if "." in key:
                # Nested update (e.g., "parameters.debug")
                parts = key.split(".")
                target = config
                for part in parts[:-1]:
                    target = getattr(target, part)
                setattr(target, parts[-1], value)
            else:
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Save
        self._save_agent_configs()
        
        # Record history
        if record_history:
            self._record_config_change(
                agent_name,
                "after",
                config.to_dict()
            )
        
        log_step(f"Updated configuration for {agent_name}")
        return True
    
    def apply_profile(self, profile_name: str) -> bool:
        """Apply a configuration profile"""
        if profile_name not in self.config_profiles:
            log_step(f"Profile '{profile_name}' not found", level="warning")
            return False
        
        profile = self.config_profiles[profile_name]
        log_step(f"Applying profile: {profile_name}")
        
        # Apply global settings
        if "global" in profile:
            for key, value in profile["global"].items():
                if "." in key:
                    # Nested key
                    parts = key.split(".")
                    target = self.global_config
                    for part in parts[:-1]:
                        if part not in target:
                            target[part] = {}
                        target = target[part]
                    target[parts[-1]] = value
                else:
                    self.global_config[key] = value
            
            # Save global config
            with open(self.global_config_file, 'w') as f:
                yaml.dump(self.global_config, f, default_flow_style=False)
        
        # Apply agent settings
        if "agents" in profile:
            for agent_pattern, settings in profile["agents"].items():
                if agent_pattern == "*":
                    # Apply to all agents
                    for agent_name in self.agent_configs:
                        self.update_agent_config(agent_name, settings, record_history=False)
                else:
                    # Apply to specific agent
                    if agent_pattern in self.agent_configs:
                        self.update_agent_config(agent_pattern, settings, record_history=False)
        
        log_step(f"Profile '{profile_name}' applied successfully")
        return True
    
    def create_profile(self, profile_name: str, profile_data: Dict[str, Any]) -> bool:
        """Create a new configuration profile"""
        profile_file = self.profiles_dir / f"{profile_name}.yaml"
        
        if profile_file.exists():
            log_step(f"Profile '{profile_name}' already exists", level="warning")
            return False
        
        with open(profile_file, 'w') as f:
            yaml.dump(profile_data, f, default_flow_style=False)
        
        self.config_profiles[profile_name] = profile_data
        log_step(f"Created profile: {profile_name}")
        return True
    
    def export_config(self, export_path: Path) -> bool:
        """Export all configurations"""
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "global_config": self.global_config,
            "agent_configs": {name: config.to_dict() for name, config in self.agent_configs.items()},
            "profiles": self.config_profiles
        }
        
        with open(export_path, 'w') as f:
            yaml.dump(export_data, f, default_flow_style=False)
        
        log_step(f"Configuration exported to: {export_path}")
        return True
    
    def import_config(self, import_path: Path, merge: bool = False) -> bool:
        """Import configurations"""
        if not import_path.exists():
            log_step(f"Import file not found: {import_path}", level="error")
            return False
        
        with open(import_path, 'r') as f:
            import_data = yaml.safe_load(f)
        
        if not merge:
            # Replace existing configs
            self.global_config = import_data.get("global_config", {})
            self.agent_configs = {}
            for name, data in import_data.get("agent_configs", {}).items():
                self.agent_configs[name] = AgentConfig.from_dict(data)
            self.config_profiles = import_data.get("profiles", {})
        else:
            # Merge with existing
            self.global_config.update(import_data.get("global_config", {}))
            for name, data in import_data.get("agent_configs", {}).items():
                self.agent_configs[name] = AgentConfig.from_dict(data)
            self.config_profiles.update(import_data.get("profiles", {}))
        
        # Save all
        with open(self.global_config_file, 'w') as f:
            yaml.dump(self.global_config, f, default_flow_style=False)
        self._save_agent_configs()
        
        log_step(f"Configuration imported from: {import_path}")
        return True
    
    def _record_config_change(self, agent_name: str, change_type: str, config_data: Dict[str, Any]):
        """Record configuration change in history"""
        change_record = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "type": change_type,
            "config": config_data
        }
        
        with open(self.history_file, 'a') as f:
            f.write(json.dumps(change_record) + '\n')
    
    def get_config_history(self, agent_name: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get configuration change history"""
        history = []
        
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                for line in f:
                    record = json.loads(line)
                    if agent_name is None or record["agent"] == agent_name:
                        history.append(record)
        
        # Return most recent changes
        return history[-limit:]
    
    def validate_config(self, agent_name: str) -> Dict[str, Any]:
        """Validate agent configuration"""
        if agent_name not in self.agent_configs:
            return {"valid": False, "errors": [f"Agent '{agent_name}' not found"]}
        
        config = self.agent_configs[agent_name]
        errors = []
        warnings = []
        
        # Check resource limits
        if config.resource_limits:
            if config.resource_limits.get("timeout", 0) < 10:
                errors.append("Timeout must be at least 10 seconds")
            if config.resource_limits.get("memory_mb", 0) < 64:
                warnings.append("Memory limit below 64MB may cause issues")
        
        # Check dependencies
        if config.dependencies:
            # Could check if dependencies are installed
            pass
        
        # Check parameters
        if not config.parameters:
            warnings.append("No parameters configured")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def generate_config_report(self) -> str:
        """Generate configuration report"""
        report = f"""# Agent Configuration Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Global Configuration

**Environment**: {self.global_config.get('environment', 'unknown')}
**Log Level**: {self.global_config.get('log_level', 'INFO')}

### Features
"""
        
        for feature, enabled in self.global_config.get('features', {}).items():
            status = "✅" if enabled else "❌"
            report += f"- {feature}: {status}\n"
        
        report += "\n## Agent Configurations\n\n"
        
        for agent_name, config in sorted(self.agent_configs.items()):
            status = "🟢" if config.enabled else "🔴"
            report += f"### {status} {agent_name} (v{config.version})\n\n"
            
            # Parameters
            if config.parameters:
                report += "**Parameters**:\n"
                for key, value in config.parameters.items():
                    report += f"- {key}: `{value}`\n"
                report += "\n"
            
            # Resource limits
            if config.resource_limits:
                report += "**Resource Limits**:\n"
                for key, value in config.resource_limits.items():
                    report += f"- {key}: {value}\n"
                report += "\n"
            
            # Validation
            validation = self.validate_config(agent_name)
            if validation["errors"]:
                report += "**⚠️ Errors**:\n"
                for error in validation["errors"]:
                    report += f"- {error}\n"
                report += "\n"
            
            if validation["warnings"]:
                report += "**⚡ Warnings**:\n"
                for warning in validation["warnings"]:
                    report += f"- {warning}\n"
                report += "\n"
        
        report += "## Available Profiles\n\n"
        for profile_name in sorted(self.config_profiles.keys()):
            report += f"- **{profile_name}**\n"
        
        return report


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Agent Configuration Manager"
    )
    parser.add_argument(
        "--get",
        help="Get configuration for agent"
    )
    parser.add_argument(
        "--set",
        nargs=2,
        metavar=("AGENT", "KEY=VALUE"),
        help="Set configuration value"
    )
    parser.add_argument(
        "--profile",
        help="Apply configuration profile"
    )
    parser.add_argument(
        "--export",
        help="Export configuration to file"
    )
    parser.add_argument(
        "--import",
        help="Import configuration from file"
    )
    parser.add_argument(
        "--validate",
        help="Validate agent configuration"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate configuration report"
    )
    parser.add_argument(
        "--history",
        help="Show configuration history for agent"
    )
    
    args = parser.parse_args()
    
    manager = ConfigManager()
    
    if args.get:
        config = manager.get_agent_config(args.get)
        if config:
            print(json.dumps(config.to_dict(), indent=2))
        else:
            print(f"Agent '{args.get}' not found")
    
    elif args.set:
        agent, key_value = args.set
        key, value = key_value.split("=", 1)
        
        # Try to parse value
        try:
            value = json.loads(value)
        except:
            pass  # Keep as string
        
        if manager.update_agent_config(agent, {key: value}):
            print(f"Updated {agent}.{key} = {value}")
        else:
            print("Failed to update configuration")
    
    elif args.profile:
        if manager.apply_profile(args.profile):
            print(f"Applied profile: {args.profile}")
        else:
            print("Failed to apply profile")
    
    elif args.export:
        if manager.export_config(Path(args.export)):
            print(f"Configuration exported to: {args.export}")
    
    elif args.validate:
        result = manager.validate_config(args.validate)
        if result["valid"]:
            print(f"✅ Configuration for {args.validate} is valid")
        else:
            print(f"❌ Configuration for {args.validate} has errors:")
            for error in result["errors"]:
                print(f"  - {error}")
        
        if result["warnings"]:
            print("\nWarnings:")
            for warning in result["warnings"]:
                print(f"  - {warning}")
    
    elif args.report:
        report = manager.generate_config_report()
        print(report)
        
        # Save report
        report_path = manager.config_dir / f"config_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
    
    elif args.history:
        history = manager.get_config_history(args.history)
        if history:
            print(f"Configuration history for {args.history}:")
            for record in history:
                print(f"\n[{record['timestamp']}] {record['type']}:")
                # Show diff if we have before/after
        else:
            print("No history found")
    
    else:
        print("Configuration Manager - Usage:")
        print("  --get AGENT           : Get agent configuration")
        print("  --set AGENT KEY=VALUE : Set configuration value")
        print("  --profile NAME        : Apply configuration profile")
        print("  --export FILE         : Export all configurations")
        print("  --import FILE         : Import configurations")
        print("  --validate AGENT      : Validate agent configuration")
        print("  --report              : Generate configuration report")
        print("  --history AGENT       : Show configuration history")


if __name__ == "__main__":
    main()