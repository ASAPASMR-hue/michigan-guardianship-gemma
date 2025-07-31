#!/usr/bin/env python3
"""
eval_rubric.py - Evaluation Rubric with CSV Integration for Michigan Guardianship AI
Implements automated evaluation of LLM responses against ground truth answers
"""

import os
import sys
import yaml
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.log_step import log_step
from scripts.validator_setup import ResponseValidator
from scripts.retrieval_setup import HybridRetriever

# Configuration paths
CONFIG_DIR = Path(__file__).parent.parent / "config"
DATA_DIR = Path(__file__).parent.parent / "data"
RUBRICS_DIR = Path(__file__).parent.parent / "rubrics"
RESULTS_DIR = Path(__file__).parent.parent / "results"

class EvaluationRubric:
    """Evaluates LLM responses against ground truth rubrics"""
    
    def __init__(self):
        # Load configurations
        self.load_configs()
        
        # Set USE_SMALL_MODEL environment variable if needed
        if os.getenv('USE_SMALL_MODEL', 'false').lower() == 'true':
            print("Using small models for development/testing")
        
        # Initialize components
        self.validator = ResponseValidator()
        self.retriever = HybridRetriever()
        
        # Create results directory
        RESULTS_DIR.mkdir(exist_ok=True)
        
        # Load QA dataset
        self.load_qa_dataset()
        
        # Load rubrics
        self.load_rubrics()
    
    def load_configs(self):
        """Load configuration files"""
        rubric_path = RUBRICS_DIR / "eval_rubric.yaml"
        if rubric_path.exists():
            with open(rubric_path, "r") as f:
                self.grading_config = yaml.safe_load(f).get("evaluation_rubric", {})
        else:
            # Fallback to default config
            self.grading_config = {
                "procedural_accuracy": {"weight": 2.5},
                "substantive_legal_accuracy": {"weight": 2.0},
                "actionability": {"weight": 2.0},
                "mode_effectiveness": {"weight": 1.5},
                "strategic_caution": {"weight": 0.5},
                "citation_quality": {"weight": 0.5},
                "harm_prevention": {"weight": 0.5}
            }
    
    def load_qa_dataset(self):
        """Load the QA CSV dataset"""
        csv_path = DATA_DIR / "guardianship_qa_cleaned.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found. Using sample data.")
            # Create sample data for testing
            self.qa_df = pd.DataFrame([
                {
                    "id": "GAP001", 
                    "question_text": "What is the filing fee for guardianship?",
                    "category": "filing",
                    "ground_truth": "The filing fee is $175. Fee waiver available with Form MC 20."
                },
                {
                    "id": "GAP002",
                    "question_text": "When are guardianship hearings held?",
                    "category": "hearings",
                    "ground_truth": "Hearings are held on Thursdays at 9:00 AM."
                }
            ])
        else:
            self.qa_df = pd.read_csv(csv_path)
            print(f"Loaded {len(self.qa_df)} questions from CSV")
    
    def load_rubrics(self):
        """Load evaluation rubrics"""
        rubric_path = RUBRICS_DIR / "rubric.yaml"
        if rubric_path.exists():
            with open(rubric_path, "r") as f:
                self.rubrics = yaml.safe_load(f)
        else:
            # Default rubric structure
            self.rubrics = {
                "evaluation_criteria": {
                    "accuracy": {
                        "weight": 0.4,
                        "description": "Factual correctness and legal accuracy"
                    },
                    "completeness": {
                        "weight": 0.3,
                        "description": "Covers all required concepts"
                    },
                    "clarity": {
                        "weight": 0.2,
                        "description": "Clear and understandable"
                    },
                    "citations": {
                        "weight": 0.1,
                        "description": "Proper legal citations"
                    }
                },
                "question_overrides": {}
            }
    
    def extract_key_facts(self, text: str) -> List[str]:
        """Extract key facts from text"""
        facts = []
        
        # Extract monetary amounts
        money_pattern = r"\$\d+(?:,\d{3})*(?:\.\d{2})?"
        facts.extend(re.findall(money_pattern, text))
        
        # Extract forms
        form_pattern = r"(?:Form\s+)?[PM]C\s*\d+"
        facts.extend(re.findall(form_pattern, text, re.IGNORECASE))
        
        # Extract days/dates
        day_pattern = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
        facts.extend(re.findall(day_pattern, text, re.IGNORECASE))
        
        time_pattern = r"\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?"
        facts.extend(re.findall(time_pattern, text))
        
        # Extract statutes
        statute_pattern = r"MCL\s+\d+\.\d+"
        facts.extend(re.findall(statute_pattern, text))
        
        return facts
    
    def calculate_accuracy_score(self, response: str, ground_truth: str, 
                               question_id: str) -> float:
        """Calculate accuracy score based on key facts matching"""
        response_facts = set(self.extract_key_facts(response))
        truth_facts = set(self.extract_key_facts(ground_truth))
        
        # Check question-specific required facts
        if question_id in self.rubrics.get("question_overrides", {}):
            override = self.rubrics["question_overrides"][question_id]
            required_facts = set()
            
            if "required_concepts" in override:
                required_facts.update(override["required_concepts"])
            if "required_citations" in override:
                required_facts.update(override["required_citations"])
            
            # Check how many required facts are present
            if required_facts:
                facts_present = sum(1 for fact in required_facts 
                                  if fact.lower() in response.lower())
                return facts_present / len(required_facts)
        
        # Fallback to fact matching
        if not truth_facts:
            return 1.0 if response else 0.0
        
        facts_found = len(response_facts.intersection(truth_facts))
        return facts_found / len(truth_facts)
    
    def calculate_completeness_score(self, response: str, question_id: str) -> float:
        """Calculate completeness based on required concepts"""
        if question_id not in self.rubrics.get("question_overrides", {}):
            # Basic length check
            return min(len(response.split()) / 50, 1.0)
        
        override = self.rubrics["question_overrides"][question_id]
        required_concepts = override.get("required_concepts", [])
        
        if not required_concepts:
            return 1.0
        
        concepts_found = sum(1 for concept in required_concepts
                           if concept.lower() in response.lower())
        
        return concepts_found / len(required_concepts)
    
    def calculate_clarity_score(self, response: str) -> float:
        """Calculate clarity score based on structure and readability"""
        score = 1.0
        
        # Check for clear structure
        if len(response) > 100:
            # Penalize very long sentences
            sentences = response.split('.')
            avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
            if avg_sentence_length > 30:
                score *= 0.8
            
            # Reward paragraph breaks
            if '\n\n' in response:
                score *= 1.1
            
            # Reward numbered/bulleted lists
            if re.search(r'^\s*\d+\.|\n\s*[-•]', response, re.MULTILINE):
                score *= 1.1
        
        return min(score, 1.0)
    
    def calculate_citation_score(self, response: str) -> float:
        """Calculate citation score"""
        # Count citations
        citation_patterns = [
            r"\(MCL \d+\.\d+\)",
            r"\(Form [PM]C \d+\)",
            r"\(see:",
            r"\(Document \d+:",
            r"\(.*Guidelines?\)"
        ]
        
        citation_count = sum(len(re.findall(pattern, response)) 
                           for pattern in citation_patterns)
        
        # Count claims needing citations
        claim_patterns = [
            r"must|shall|required to",
            r"MCL \d+\.\d+",
            r"Form [PM]C \d+",
            r"\$\d+"
        ]
        
        claim_count = sum(len(re.findall(pattern, response, re.IGNORECASE))
                         for pattern in claim_patterns)
        
        if claim_count == 0:
            return 1.0
        
        return min(citation_count / claim_count, 1.0)
    
    def evaluate_response(self, question_id: str, question_text: str,
                         response: str, ground_truth: str,
                         retrieved_chunks: List[str]) -> Dict:
        """Evaluate a single response"""
        
        # Get question type for validation
        question_type = self._classify_question_type(question_text)
        
        # Run validation
        validation_result = self.validator.validate(
            response, retrieved_chunks, question_type, question_text
        )
        
        # If out of scope, special handling
        if validation_result.get("out_of_scope"):
            return {
                "question_id": question_id,
                "pass": True,
                "out_of_scope": True,
                "scores": {"overall": 1.0},
                "validation": validation_result
            }
        
        # If validation failed, return early
        if not validation_result["pass"]:
            return {
                "question_id": question_id,
                "pass": False,
                "reason": validation_result.get("reason"),
                "scores": {"overall": 0.0},
                "validation": validation_result
            }
        
        # Calculate component scores
        scores = {
            "accuracy": self.calculate_accuracy_score(response, ground_truth, question_id),
            "completeness": self.calculate_completeness_score(response, question_id),
            "clarity": self.calculate_clarity_score(response),
            "citations": self.calculate_citation_score(response),
            "hallucination": 1.0 - validation_result["scores"]["hallucination"],
            "mode_effectiveness": validation_result["scores"]["mode_effectiveness"]
        }
        
        # Calculate weighted overall score
        # Map old score names to new rubric criteria names
        score_mapping = {
            "accuracy": "procedural_accuracy",
            "completeness": "actionability",
            "clarity": "mode_effectiveness",
            "citations": "citation_quality"
        }
        
        overall_score = 0.0
        total_weight = 0.0
        
        for old_name, new_name in score_mapping.items():
            if new_name in self.grading_config and old_name in scores:
                weight = self.grading_config[new_name].get("weight", 0)
                overall_score += scores[old_name] * weight
                total_weight += weight
        
        # Add scores for criteria not in the mapping
        if "hallucination" in scores and "harm_prevention" in self.grading_config:
            weight = self.grading_config["harm_prevention"].get("weight", 0)
            overall_score += scores["hallucination"] * weight
            total_weight += weight
            
        # Normalize to account for actual weights used
        if total_weight > 0:
            overall_score = overall_score / total_weight
        
        scores["overall"] = overall_score
        
        return {
            "question_id": question_id,
            "pass": overall_score >= self.grading_config.get("passing_threshold", 0.7),
            "scores": scores,
            "validation": validation_result
        }
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type for validation"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["fee", "cost", "form", "address"]):
            return "simple"
        elif any(word in question_lower for word in ["icwa", "tribal", "emergency"]):
            return "complex"
        else:
            return "standard"
    
    def evaluate_dataset(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Evaluate entire dataset or sample"""
        if sample_size:
            eval_df = self.qa_df.sample(min(sample_size, len(self.qa_df)))
        else:
            eval_df = self.qa_df
        
        results = []
        
        for _, row in eval_df.iterrows():
            print(f"\nEvaluating {row['id']}: {row['question_text'][:50]}...")
            
            # Retrieve relevant chunks
            retrieved_docs, metadata = self.retriever.retrieve(row['question_text'])
            chunks = [doc['document'] for doc in retrieved_docs]
            
            # Generate mock response for testing (in production, this would be from LLM)
            # For now, use ground truth with some modifications
            mock_response = self._generate_mock_response(row, chunks)
            
            # Evaluate
            eval_result = self.evaluate_response(
                row['id'],
                row['question_text'],
                mock_response,
                row.get('ground_truth', ''),
                chunks
            )
            
            results.append({
                'question_id': row['id'],
                'category': row.get('category', 'unknown'),
                'pass': eval_result['pass'],
                'overall_score': eval_result['scores']['overall'],
                **{f'score_{k}': v for k, v in eval_result['scores'].items() if k != 'overall'}
            })
        
        results_df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = RESULTS_DIR / f"evaluation_results_{timestamp}.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")
        
        # Print summary
        self._print_summary(results_df)
        
        return results_df
    
    def _generate_mock_response(self, row: pd.Series, chunks: List[str]) -> str:
        """Generate mock response for testing"""
        # In production, this would be replaced with actual LLM call
        if 'ground_truth' in row and pd.notna(row['ground_truth']):
            # Add some structure to ground truth
            response = row['ground_truth']
            
            # Add a citation if there's a statute
            if "MCL" in response and "(" not in response:
                response = re.sub(r"(MCL \d+\.\d+)", r"\1 (Michigan Compiled Laws)", response)
            
            # Add disclaimer for complex questions
            if "?" in row['question_text'] and len(row['question_text']) > 50:
                response += "\n\nNote: This is general information about Michigan guardianship procedures. For advice specific to your situation, please consult with a licensed Michigan attorney."
            
            return response
        else:
            # Generate basic response from chunks
            if chunks:
                return f"Based on Michigan guardianship law: {chunks[0][:200]}..."
            else:
                return "I can help with that guardianship question."
    
    def _print_summary(self, results_df: pd.DataFrame):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        print(f"\nTotal Questions Evaluated: {len(results_df)}")
        print(f"Passed: {results_df['pass'].sum()} ({results_df['pass'].mean()*100:.1f}%)")
        
        print("\nAverage Scores:")
        score_columns = [col for col in results_df.columns if col.startswith('score_')]
        for col in score_columns:
            metric = col.replace('score_', '').replace('_', ' ').title()
            print(f"  {metric}: {results_df[col].mean():.3f}")
        
        print(f"\nOverall Average: {results_df['overall_score'].mean():.3f}")
        
        # Category breakdown
        if 'category' in results_df.columns:
            print("\nBy Category:")
            category_stats = results_df.groupby('category').agg({
                'pass': 'mean',
                'overall_score': 'mean'
            })
            for category, stats in category_stats.iterrows():
                print(f"  {category}: {stats['pass']*100:.1f}% pass, {stats['overall_score']:.3f} avg score")

def main():
    """Main execution function"""
    log_step("Starting evaluation rubric", "Initializing evaluation system", "Per Part A.6")
    
    evaluator = EvaluationRubric()
    
    # Run evaluation on sample
    print("\nRunning evaluation on sample questions...")
    results = evaluator.evaluate_dataset(sample_size=5)
    
    log_step("Evaluation complete", "Generated evaluation results", "Quality assurance")
    
    print("\n✓ Evaluation rubric setup completed successfully!")

if __name__ == "__main__":
    main()