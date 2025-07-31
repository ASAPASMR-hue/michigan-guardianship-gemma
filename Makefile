.PHONY: extract-configs phase1 clean-chroma test-phase1 clean-all

extract-configs:
	python scripts/split_playbook.py

# Phase 1: Document Embedding, Retrieval, and Validation
phase1: extract-configs
	@echo "=== Phase 1: Building Michigan Guardianship AI Foundation ==="
	@echo "[$(shell date +'%Y-%m-%d %H:%M:%S')] Starting Phase 1 pipeline..." | tee -a logs/phase1_pipeline.log
	@echo "\n1. Embedding knowledge base documents..."
	@python scripts/embed_kb.py 2>&1 | tee -a logs/phase1_pipeline.log
	@echo "\n2. Setting up hybrid retrieval system..."
	@python scripts/retrieval_setup.py 2>&1 | tee -a logs/phase1_pipeline.log
	@echo "\n3. Configuring response validation..."
	@python scripts/validator_setup.py 2>&1 | tee -a logs/phase1_pipeline.log
	@echo "\n4. Running evaluation tests..."
	@python scripts/eval_rubric.py 2>&1 | tee -a logs/phase1_pipeline.log
	@echo "\n[$(shell date +'%Y-%m-%d %H:%M:%S')] ✓ Phase 1 completed successfully!" | tee -a logs/phase1_pipeline.log
	@echo "Full log saved to: logs/phase1_pipeline.log"

# Clean ChromaDB for fresh start
clean-chroma:
	@echo "Removing ChromaDB data..."
	rm -rf chroma_db/
	@echo "ChromaDB cleaned."

# Run tests with small models
test-phase1:
	@echo "Running Phase 1 tests with small models..."
	@export USE_SMALL_MODEL=true && \
	make phase1

# Clean all generated files
clean-all: clean-chroma
	@echo "Cleaning all generated files..."
	rm -rf logs/*.log results/*.csv __pycache__ scripts/__pycache__
	@echo "Cleanup complete."

# Phase 3 testing commands
phase3-setup:
	@echo "Setting up Phase 3 testing environment..."
	./setup_phase3.sh

phase3-test:
	@echo "Testing Phase 3 setup..."
	python scripts/test_phase3_setup.py

phase3-run:
	@echo "Running Phase 3 full evaluation..."
	python scripts/run_full_evaluation.py

phase3-analyze:
	@echo "Analyzing Phase 3 results..."
	@echo "Usage: make phase3-analyze RUN_ID=run_20250128_1430"
	python scripts/analyze_results.py --run_id $(RUN_ID)