#!/bin/bash
# Backup Phase 1 configuration and scripts

BACKUP_DIR="../phase1-backup-$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Copy essential files
cp -r config/ "$BACKUP_DIR/"
cp -r scripts/migrate_to_pinecone.py "$BACKUP_DIR/"
cp -r scripts/setup_pinecone.py "$BACKUP_DIR/"
cp .env "$BACKUP_DIR/"
cp PHASE1_COMPLETION_REPORT.md "$BACKUP_DIR/"

echo "✅ Backup created at: $BACKUP_DIR"
echo "Files backed up:"
ls -la "$BACKUP_DIR"
