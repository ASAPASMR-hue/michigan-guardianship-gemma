#!/usr/bin/env python3
"""
Phase 3a: Full Evaluation Runner
Generates raw test data for all models without evaluation
"""

import os
import sys
import json
import yaml
import subprocess
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.llm_handler import LLMHandler, sanitize_model_name
from scripts.adaptive_retrieval import AdaptiveHybridRetriever
from scripts.logger import log_step


class FullEvaluationRunner:
    """Runs complete evaluation across all models"""
    
    def __init__(self, config_path: str = "config/model_configs_phase3.yaml"):
        """Initialize evaluation runner"""
        self.start_time = datetime.now()
        self.run_id = f"run_{self.start_time.strftime('%Y%m%d_%H%M')}"
        
        # Load configurations
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models = self.config['models']
        self.testing_config = self.config['testing']
        
        # Initialize components
        self.llm_handler = LLMHandler(timeout=self.testing_config['timeout'])
        self.retrieval = AdaptiveHybridRetriever()
        
        # Load system prompts
        self.system_prompt = self._load_system_prompt()
        
        # Create output directory
        self.output_dir = Path(f"results/{self.run_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track failed models
        self.failed_models = []
        
        log_step(f"Initialized evaluation runner with run_id: {self.run_id}")
    
    def _load_system_prompt(self) -> str:
        """Load master system prompt"""
        prompt_path = Path("kb_files/Instructive/Master Prompt Template.txt")
        if prompt_path.exists():
            return prompt_path.read_text()
        else:
            log_step("Warning: Master prompt not found, using default")
            return "You are a Michigan minor guardianship assistant serving Genesee County residents."
    
    def _get_git_info(self) -> Dict[str, Any]:
        """Get git commit information"""
        try:
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD']
            ).strip().decode('utf-8')
            
            git_status = subprocess.check_output(
                ['git', 'status', '--porcelain']
            ).decode('utf-8')
            
            has_uncommitted = bool(git_status.strip())
            
            return {
                "commit_hash": git_hash,
                "has_uncommitted_changes": has_uncommitted
            }
        except:
            return {
                "commit_hash": "unknown",
                "has_uncommitted_changes": False
            }
    
    def _save_metadata(self, questions_count: int):
        """Save run metadata"""
        git_info = self._get_git_info()
        
        metadata = {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "git_commit_hash": git_info["commit_hash"],
            "has_uncommitted_changes": git_info["has_uncommitted_changes"],
            "models": [m['id'] for m in self.models],
            "total_questions": questions_count,
            "configuration": {
                "timeout": self.testing_config['timeout'],
                "retrieval_top_k": self.testing_config['retrieval_top_k'],
                "system_prompt_version": self.testing_config['system_prompt_version'],
                "embedding_model": "BAAI/bge-m3",
                "temperature": 0.1,
                "max_tokens": 2000
            },
            "environment": {
                "python_version": sys.version,
                "openrouter_api": "https://openrouter.ai/api/v1",
                "google_ai_api": "generativelanguage.googleapis.com"
            }
        }
        
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        log_step(f"Saved metadata to {self.output_dir}/metadata.json")
    
    def load_questions(self) -> pd.DataFrame:
        """Load test questions from CSV"""
        # Use the new file as specified in the prompt
        csv_path = Path("/Users/claytoncanady/Downloads/active_workspace/michigan-guardianship-ai/data/Synthetic Test Questions (2).xlsx")
        
        if not csv_path.exists():
            # Fallback to original path
            csv_path = Path("data/Synthetic Test Questions.xlsx")
            if not csv_path.exists():
                csv_path = Path("/Users/claytoncanady/Library/michigan-guardianship-ai/Synthetic Test Questions - Sheet1.csv")
        
        if csv_path.suffix == '.xlsx':
            df = pd.read_excel(csv_path)
        else:
            df = pd.read_csv(csv_path)
        
        log_step(f"Loaded {len(df)} questions from {csv_path}")
        return df
    
    def process_question(
        self,
        question: Dict[str, Any],
        model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single question for a model"""
        start_time = time.time()
        
        try:
            # Retrieve context
            # Note: HybridRetriever determines complexity internally
            documents, metadata = self.retrieval.retrieve(question['question'])
            
            # Convert to expected format
            retrieval_result = {
                'documents': documents,
                'metadata': metadata
            }
            
            # Build prompt
            context_text = "\n\n".join([
                f"[Source: {doc.get('metadata', {}).get('source', 'Unknown')}]\n{doc.get('content', doc.get('page_content', ''))}"
                for doc in documents
            ])
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""Context from Michigan Minor Guardianship Knowledge Base:

{context_text}

Question: {question['question']}

Please provide a comprehensive answer based on the context provided."""}
            ]
            
            # Call LLM
            llm_result = self.llm_handler.call_llm(
                model_id=model['id'],
                messages=messages,
                model_api='google_ai' if model['api'] == 'google' else model['api'],
                temperature=model.get('temperature', 0.1),
                max_tokens=model.get('max_tokens', 2000)
            )
            
            # Build result
            result = {
                "run_id": self.run_id,
                "question_id": question['id'],
                "question_text": question['question'],
                "category": question.get('category', ''),
                "subcategory": question.get('subcategory', ''),
                "complexity_tier": question.get('complexity_tier', 'standard'),
                "retrieved_context": [
                    {
                        "content": doc.get('content', doc.get('page_content', '')),
                        "metadata": doc.get('metadata', {}),
                        "score": doc.get('score', 0.0)
                    }
                    for doc in documents
                ],
                "model": model['id'],
                "response": llm_result['response'],
                "latency": llm_result['latency'],
                "cost_usd": llm_result['cost_usd'],
                "timestamp": datetime.now().isoformat(),
                "error": llm_result['error']
            }
            
            return result
            
        except Exception as e:
            log_step(f"Error processing question {question['id']}: {str(e)}", level="error")
            return {
                "run_id": self.run_id,
                "question_id": question['id'],
                "question_text": question['question'],
                "model": model['id'],
                "response": "",
                "latency": time.time() - start_time,
                "cost_usd": 0,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def test_model(self, model: Dict[str, Any], questions: pd.DataFrame) -> bool:
        """Test a single model against all questions"""
        log_step(f"\n{'='*60}")
        log_step(f"Testing model: {model['id']}")
        log_step(f"{'='*60}")
        
        # Test connectivity first
        log_step("Testing model connectivity...")
        test_result = self.llm_handler.test_connectivity(model['id'], model['api'])
        
        if not test_result['success']:
            error_msg = f"Model {model['id']} failed connectivity test: {test_result['error']}"
            log_step(error_msg, level="error")
            self.failed_models.append({
                "model": model['id'],
                "error": test_result['error'],
                "timestamp": datetime.now().isoformat()
            })
            return False
        
        log_step("✅ Connectivity test passed")
        
        # Process questions
        results = []
        batch_size = self.testing_config.get('batch_size', 10)
        
        for idx, row in questions.iterrows():
            try:
                question = row.to_dict()
                result = self.process_question(question, model)
                results.append(result)
                
                # Progress update
                if (idx + 1) % batch_size == 0:
                    log_step(f"  Processed {idx + 1}/{len(questions)} questions...")
                
            except Exception as e:
                log_step(f"  ⚠️  Error on question {row['id']}: {str(e)}", level="warning")
                results.append({
                    "run_id": self.run_id,
                    "question_id": row['id'],
                    "model": model['id'],
                    "response": "",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Save results
        safe_name = sanitize_model_name(model['id'])
        output_path = self.output_dir / f"{safe_name}_results.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        log_step(f"✅ Saved results for {model['id']} to {output_path}")
        return True
    
    def run(self):
        """Run full evaluation"""
        log_step("Starting Phase 3a Full Evaluation")
        
        # Load questions
        questions = self.load_questions()
        
        # Limit questions if specified
        max_questions = self.testing_config.get('max_questions')
        if max_questions and max_questions < len(questions):
            questions = questions.head(max_questions)
            log_step(f"Limited to {max_questions} questions for testing")
        
        # Save metadata
        self._save_metadata(len(questions))
        
        # Test each model
        successful_models = 0
        
        for model in self.models:
            try:
                if self.test_model(model, questions):
                    successful_models += 1
            except Exception as e:
                log_step(f"❌ Critical error with model {model['id']}: {str(e)}", level="error")
                self.failed_models.append({
                    "model": model['id'],
                    "error": f"Critical failure: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Save failure log if any
        if self.failed_models:
            with open(self.output_dir / "failed_models.json", 'w') as f:
                json.dump(self.failed_models, f, indent=2)
            log_step(f"\n⚠️  {len(self.failed_models)} models failed. See failed_models.json")
        
        # Final summary
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        summary = {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "total_models": len(self.models),
            "successful_models": successful_models,
            "failed_models": len(self.failed_models),
            "total_questions": len(questions)
        }
        
        with open(self.output_dir / "run_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        log_step(f"\n{'='*60}")
        log_step(f"✅ Evaluation Complete!")
        log_step(f"Run ID: {self.run_id}")
        log_step(f"Duration: {duration/60:.1f} minutes")
        log_step(f"Successfully tested {successful_models}/{len(self.models)} models")
        log_step(f"Results saved to: {self.output_dir}")
        log_step(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Phase 3 evaluation")
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    parser.add_argument("--questions", type=int, help="Number of questions to test")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    runner = FullEvaluationRunner()
    
    # Override models if specified
    if args.models:
        runner.models = [m for m in runner.models if m['id'] in args.models]
    
    # Limit questions if specified
    if args.questions:
        runner.testing_config['max_questions'] = args.questions
    
    # Update output directory
    if args.output_dir:
        runner.output_dir = Path(args.output_dir) / runner.run_id
        runner.output_dir.mkdir(parents=True, exist_ok=True)
    
    runner.run()