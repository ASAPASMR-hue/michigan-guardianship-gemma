#!/usr/bin/env python3
"""
Quick Start Guide for Michigan Guardianship AI Agents
Interactive setup and tutorial for new users
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.log_step import log_step
from agent_utils import AgentUtils


class QuickStart:
    """Interactive quick start guide"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.completed_steps = set()
        
    def run(self):
        """Run the interactive quick start"""
        self.print_header()
        
        while True:
            choice = self.show_menu()
            
            if choice == '1':
                self.check_environment()
            elif choice == '2':
                self.setup_agents()
            elif choice == '3':
                self.run_first_analysis()
            elif choice == '4':
                self.explore_dashboards()
            elif choice == '5':
                self.setup_automation()
            elif choice == '6':
                self.show_advanced_features()
            elif choice == '7':
                self.run_diagnostics()
            elif choice == '8':
                self.show_cheat_sheet()
            elif choice == '9':
                print("\nThank you for using Michigan Guardianship AI!")
                break
            else:
                print("\nInvalid choice. Please try again.")
            
            input("\nPress Enter to continue...")
    
    def print_header(self):
        """Print welcome header"""
        print("""
╔══════════════════════════════════════════════════════╗
║     Michigan Guardianship AI - Agent Quick Start     ║
╚══════════════════════════════════════════════════════╝

Welcome! This guide will help you get started with the
intelligent agent ecosystem.
        """)
    
    def show_menu(self) -> str:
        """Show main menu"""
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        
        print("\nGetting Started:")
        print(f"  1. {'✓' if '1' in self.completed_steps else ' '} Check Environment")
        print(f"  2. {'✓' if '2' in self.completed_steps else ' '} Setup Agents")
        print(f"  3. {'✓' if '3' in self.completed_steps else ' '} Run First Analysis")
        
        print("\nExplore Features:")
        print(f"  4. {'✓' if '4' in self.completed_steps else ' '} Explore Dashboards")
        print(f"  5. {'✓' if '5' in self.completed_steps else ' '} Setup Automation")
        print(f"  6. {'✓' if '6' in self.completed_steps else ' '} Advanced Features")
        
        print("\nHelp & Support:")
        print("  7. Run Diagnostics")
        print("  8. Show Cheat Sheet")
        print("  9. Exit")
        
        return input("\nEnter your choice (1-9): ").strip()
    
    def check_environment(self):
        """Check environment setup"""
        print("\n" + "="*50)
        print("CHECKING ENVIRONMENT")
        print("="*50)
        
        checks = []
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major >= 3 and python_version.minor >= 8:
            checks.append(("Python 3.8+", True))
        else:
            checks.append(("Python 3.8+", False))
        
        # Check required packages
        required_packages = [
            "pandas", "numpy", "pyyaml", "plotly", "jinja2", 
            "schedule", "psutil", "matplotlib", "seaborn"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                checks.append((f"{package}", True))
            except ImportError:
                checks.append((f"{package}", False))
        
        # Check directories
        required_dirs = [
            "agent_configs",
            "monitoring", 
            "scheduler",
            "results/dashboards"
        ]
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                checks.append((f"Directory: {dir_name}", True))
            else:
                checks.append((f"Directory: {dir_name}", False))
        
        # Display results
        print("\nEnvironment Check Results:")
        print("-" * 30)
        
        all_passed = True
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"{status} {check_name}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n✅ All checks passed! Your environment is ready.")
            self.completed_steps.add('1')
        else:
            print("\n❌ Some checks failed. Please install missing dependencies:")
            print("   pip install -r requirements.txt")
            print("   make setup")
    
    def setup_agents(self):
        """Setup agent configurations"""
        print("\n" + "="*50)
        print("SETTING UP AGENTS")
        print("="*50)
        
        print("\nInitializing agent configurations...")
        
        # Run setup command
        os.chdir(Path(__file__).parent)
        result = subprocess.run(["make", "setup"], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Agent environment initialized successfully!")
            
            # Show what was created
            print("\nCreated directories:")
            for dir_name in ["agent_configs", "monitoring", "scheduler", "results/dashboards"]:
                print(f"  - {dir_name}/")
            
            print("\nDefault configurations created for:")
            agents = [
                "test_analyzer", "workflow_optimizer", "ai_expert",
                "code_refactorer", "system_architect", "product_strategist",
                "analytics_engineer", "dashboard_generator", "agent_scheduler",
                "performance_monitor"
            ]
            for agent in agents:
                print(f"  - {agent}")
            
            self.completed_steps.add('2')
        else:
            print("❌ Setup failed. Please check the error messages above.")
    
    def run_first_analysis(self):
        """Guide through first analysis"""
        print("\n" + "="*50)
        print("RUNNING YOUR FIRST ANALYSIS")
        print("="*50)
        
        # Check for existing runs
        latest_run = AgentUtils.get_latest_run_id()
        
        if not latest_run:
            print("\n⚠️  No test runs found in the results directory.")
            print("To run an analysis, you first need test results from Phase 3.")
            print("\nTry running: python scripts/run_phase3_tests.py --models 'mistralai/mistral-nemo:free'")
            return
        
        print(f"\nFound latest run: {latest_run}")
        choice = input("\nWould you like to analyze this run? (y/n): ").lower()
        
        if choice == 'y':
            print("\nRunning quick analysis pipeline...")
            print("This will:")
            print("  1. Check system health")
            print("  2. Analyze test results")
            print("  3. Generate recommendations")
            
            # Run analysis
            os.chdir(Path(__file__).parent)
            subprocess.run([
                "python", "run_agent_pipeline.py", 
                "--pipeline", "quick_check",
                "--run-id", latest_run
            ])
            
            self.completed_steps.add('3')
            
            print("\n✅ Analysis complete! Check the results above.")
    
    def explore_dashboards(self):
        """Guide through dashboard features"""
        print("\n" + "="*50)
        print("EXPLORING DASHBOARDS")
        print("="*50)
        
        dashboards_dir = self.project_root / "results" / "dashboards"
        
        if not dashboards_dir.exists() or not list(dashboards_dir.glob("*.html")):
            print("\n⚠️  No dashboards found.")
            
            latest_run = AgentUtils.get_latest_run_id()
            if latest_run:
                print(f"\nWould you like to generate a dashboard for {latest_run}?")
                choice = input("(y/n): ").lower()
                
                if choice == 'y':
                    print("\nGenerating dashboard...")
                    os.chdir(Path(__file__).parent)
                    subprocess.run([
                        "python", "dashboard_generator.py",
                        "--test-results", latest_run
                    ])
                    
                    # Create index
                    subprocess.run([
                        "python", "dashboard_generator.py",
                        "--index"
                    ])
        
        # List existing dashboards
        dashboards = list(dashboards_dir.glob("*.html"))
        if dashboards:
            print(f"\nFound {len(dashboards)} dashboard(s):")
            for dash in dashboards[-5:]:  # Show last 5
                print(f"  - {dash.name}")
            
            print("\nTo view dashboards:")
            print(f"  1. Open: {dashboards_dir / 'index.html'}")
            print("  2. Or run: make dashboard-index")
            
            self.completed_steps.add('4')
    
    def setup_automation(self):
        """Guide through automation setup"""
        print("\n" + "="*50)
        print("SETTING UP AUTOMATION")
        print("="*50)
        
        print("\nThe agent scheduler can run agents automatically.")
        print("\nDefault scheduled jobs:")
        print("  - Daily quick check (9:00 AM)")
        print("  - Weekly system review (Monday 10:00 AM)")
        print("  - Daily dashboard update (6:00 PM)")
        
        choice = input("\nWould you like to see current scheduled jobs? (y/n): ").lower()
        
        if choice == 'y':
            os.chdir(Path(__file__).parent)
            subprocess.run(["python", "agent_scheduler.py", "--list-jobs"])
        
        print("\nTo start the scheduler:")
        print("  1. Run: make schedule-start")
        print("  2. Or: python agent_scheduler.py --start --daemon")
        
        print("\nTo add custom jobs:")
        print('  python agent_scheduler.py --add-job \'{"name": "my_job", ...}\'')
        
        self.completed_steps.add('5')
    
    def show_advanced_features(self):
        """Show advanced features"""
        print("\n" + "="*50)
        print("ADVANCED FEATURES")
        print("="*50)
        
        features = {
            "Performance Monitoring": [
                "Track agent execution times",
                "Monitor resource usage",
                "Get alerts on performance issues",
                "Command: python performance_monitor.py --report"
            ],
            "Configuration Profiles": [
                "Switch between dev/prod settings",
                "Validate configurations", 
                "Track config history",
                "Command: python config_manager.py --profile production"
            ],
            "Agent Pipelines": [
                "Chain multiple agents together",
                "Predefined workflows",
                "Custom sequences",
                "Command: python run_agent_pipeline.py --pipeline comprehensive"
            ],
            "Batch Processing": [
                "Process multiple runs",
                "Parallel execution",
                "Progress tracking",
                "Use agent_utils.py helpers"
            ]
        }
        
        for feature, details in features.items():
            print(f"\n{feature}:")
            for detail in details:
                print(f"  - {detail}")
        
        self.completed_steps.add('6')
    
    def run_diagnostics(self):
        """Run system diagnostics"""
        print("\n" + "="*50)
        print("RUNNING DIAGNOSTICS")
        print("="*50)
        
        print("\nChecking system health...")
        
        # Performance monitor health
        os.chdir(Path(__file__).parent)
        subprocess.run(["python", "performance_monitor.py", "--health"])
        
        print("\nChecking agent configurations...")
        subprocess.run(["make", "validate-all"])
        
        print("\nChecking dependencies...")
        subprocess.run(["make", "check-deps"])
        
        print("\n✅ Diagnostics complete!")
    
    def show_cheat_sheet(self):
        """Show command cheat sheet"""
        print("\n" + "="*50)
        print("AGENT COMMAND CHEAT SHEET")
        print("="*50)
        
        commands = {
            "Quick Operations": [
                "make quick-check         # Fast system validation",
                "make golden-test         # Run golden questions", 
                "make health-check        # System health status"
            ],
            "Analysis": [
                "make full-analysis RUN_ID=xxx    # Complete analysis",
                "make system-review               # Architecture review",
                "make optimization RUN_ID=xxx     # AI optimization"
            ],
            "Dashboards": [
                "make dashboard RUN_ID=xxx        # Generate dashboard",
                "make dashboard-index             # Update index",
                "open results/dashboards/index.html"
            ],
            "Automation": [
                "make schedule-start              # Start scheduler",
                "make schedule-status             # Check status",
                "make monitor-demo                # Performance demo"
            ],
            "Development": [
                "make test                        # Run all tests",
                "make test-quick                  # Quick tests",
                "python agent_utils.py --latest-run  # Get latest run"
            ]
        }
        
        for category, cmds in commands.items():
            print(f"\n{category}:")
            for cmd in cmds:
                print(f"  {cmd}")
        
        print("\n💡 Tip: Use 'make help' to see all available commands")


def main():
    """Run the quick start guide"""
    quick_start = QuickStart()
    
    try:
        quick_start.run()
    except KeyboardInterrupt:
        print("\n\nQuick start interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check the logs and try again.")


if __name__ == "__main__":
    main()