#!/usr/bin/env python3
import sys
import datetime

def log_step(action, details, rationale):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("project_log.md", "a") as f:
        f.write(f"- **Timestamp**: {timestamp}\n")
        f.write(f"- **Action**: {action}\n")
        f.write(f"- **Details**: {details}\n")
        f.write(f"- **Rationale**: {rationale}\n\n")

if __name__ == "__main__":
    action = sys.argv[1]
    details = sys.argv[2]
    rationale = sys.argv[3]
    log_step(action, details, rationale)