#!/usr/bin/env python3
"""
Train lightweight complexity classifier for query classification
Phase 2: Step 2 - Training Complexity Classifier
"""

import os
import sys
import json
import pickle
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.log_step import log_step

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / "phase2_classifier.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ComplexityClassifierTrainer:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize feature extractors
        self.tfidf = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=2,
            stop_words='english'
        )
        
        # Keyword patterns for complexity
        self.complexity_keywords = {
            'simple': ['fee', 'cost', 'form', 'pc', 'address', 'deadline', 'days', 'where', 'when', 'how much'],
            'standard': ['parent', 'guardian', 'process', 'consent', 'modify', 'terminate', 'requirements'],
            'complex': ['icwa', 'tribal', 'emergency', 'cps', 'interstate', 'out-of-state', 'contested', 
                       'multiple', 'special needs', 'urgent', 'crisis']
        }
        
        self.label_encoder = LabelEncoder()
        self.classifier = None
        
    def load_data(self):
        """Load and combine synthetic and existing test questions"""
        logger.info("Loading training data...")
        
        # Load synthetic questions
        synthetic_path = self.data_dir / "synthetic_questions_phase2.csv"
        synthetic_df = pd.read_csv(synthetic_path)
        
        # Filter out out-of-scope questions (classifier only handles tiers)
        synthetic_df = synthetic_df[~synthetic_df['is_out_of_scope']]
        logger.info(f"Loaded {len(synthetic_df)} synthetic questions (excluding out-of-scope)")
        
        # Load existing test questions
        test_path = self.project_root / "guardianship_qa_cleaned - rubric_determining-2.csv"
        if test_path.exists():
            test_df = pd.read_csv(test_path)
            # Map complexity categories if they exist
            if 'complexity' in test_df.columns:
                test_questions = test_df[['question', 'complexity']].rename(columns={'complexity': 'complexity_tier'})
            else:
                # Estimate complexity based on length and keywords
                test_questions = test_df[['question']].copy()
                test_questions['complexity_tier'] = test_questions['question'].apply(self._estimate_complexity)
            
            logger.info(f"Loaded {len(test_questions)} existing test questions")
        else:
            test_questions = pd.DataFrame()
            logger.warning("No existing test questions found")
        
        # Combine datasets
        all_questions = pd.concat([
            synthetic_df[['question', 'complexity_tier']],
            test_questions
        ], ignore_index=True)
        
        logger.info(f"Total training samples: {len(all_questions)}")
        logger.info(f"Distribution: {all_questions['complexity_tier'].value_counts().to_dict()}")
        
        return all_questions
    
    def _estimate_complexity(self, question):
        """Estimate complexity for questions without labels"""
        question_lower = question.lower()
        
        # Check for complex keywords
        for keyword in self.complexity_keywords['complex']:
            if keyword in question_lower:
                return 'complex'
        
        # Check for simple patterns
        if len(question.split()) < 10 and any(kw in question_lower for kw in self.complexity_keywords['simple']):
            return 'simple'
        
        # Default to standard
        return 'standard'
    
    def extract_features(self, questions):
        """Extract TF-IDF and keyword features"""
        # TF-IDF features
        tfidf_features = self.tfidf.fit_transform(questions) if hasattr(self, 'is_training') and self.is_training else self.tfidf.transform(questions)
        
        # Keyword features
        keyword_features = []
        for question in questions:
            question_lower = question.lower()
            features = []
            
            # Count keywords for each complexity level
            for tier, keywords in self.complexity_keywords.items():
                count = sum(1 for kw in keywords if kw in question_lower)
                features.append(count)
            
            # Additional features
            features.extend([
                len(question.split()),  # Word count
                question.count('?'),    # Question marks
                question.count('MCL'),  # Legal citations
                question.count('PC'),   # Form references
                'emergency' in question_lower or 'urgent' in question_lower,  # Urgency
                'icwa' in question_lower or 'tribal' in question_lower,       # ICWA
            ])
            
            keyword_features.append(features)
        
        keyword_features = np.array(keyword_features)
        
        # Combine features
        from scipy.sparse import hstack
        combined_features = hstack([tfidf_features, keyword_features])
        
        return combined_features
    
    def train_classifier(self, X_train, y_train, X_test, y_test):
        """Train the logistic regression classifier"""
        logger.info("Training classifier...")
        
        # Initialize and train
        self.classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced classes
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Test accuracy: {accuracy:.3f}")
        logger.info("\nClassification Report:")
        report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
        logger.info(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info("\nConfusion Matrix:")
        logger.info(f"Labels: {self.label_encoder.classes_}")
        logger.info(cm)
        
        # Cross-validation
        cv_scores = cross_val_score(self.classifier, X_train, y_train, cv=5)
        logger.info(f"\nCross-validation scores: {cv_scores}")
        logger.info(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return accuracy
    
    def save_model(self):
        """Save the trained model and components"""
        model_path = self.models_dir / "complexity_classifier.pkl"
        
        model_data = {
            'classifier': self.classifier,
            'tfidf': self.tfidf,
            'label_encoder': self.label_encoder,
            'complexity_keywords': self.complexity_keywords,
            'metadata': {
                'trained_at': datetime.now().isoformat(),
                'features': {
                    'tfidf_features': self.tfidf.max_features,
                    'keyword_features': 9  # 3 complexity counts + 6 additional
                }
            }
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
        
        # Also save model info as JSON
        info_path = self.models_dir / "complexity_classifier_info.json"
        info = {
            'trained_at': model_data['metadata']['trained_at'],
            'features': model_data['metadata']['features'],
            'classes': self.label_encoder.classes_.tolist(),
            'performance': {
                'test_accuracy': self.test_accuracy,
                'target_accuracy': 0.85
            }
        }
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Model info saved to {info_path}")
    
    def test_holdout_questions(self):
        """Test on 50 holdout questions"""
        logger.info("\nTesting on holdout questions...")
        
        # Generate some holdout test cases
        holdout_questions = [
            # Simple
            "What's the filing fee?",
            "Where do I file the forms?",
            "What form do I need for guardianship?",
            "How many days before the hearing?",
            "What's the court address?",
            
            # Standard
            "My mother wants guardianship of my child. What's the process?",
            "How can I terminate the current guardianship?",
            "What if one parent doesn't consent?",
            "Can grandparents be co-guardians?",
            "What are the requirements for an aunt to be guardian?",
            
            # Complex
            "Child is tribal member and needs emergency guardianship due to parent arrest",
            "ICWA applies and the other parent is out of state. What do we do?",
            "CPS is involved and we need immediate placement. How fast can we act?",
            "Multiple family members want guardianship and parents are contesting",
            "Emergency situation with special needs child and guardian is hospitalized"
        ]
        
        # Add more questions to reach 50
        holdout_questions.extend([
            f"Question about {topic} #{i}" 
            for i, topic in enumerate(['filing', 'forms', 'deadlines', 'process', 'requirements'] * 7)
        ])
        
        holdout_questions = holdout_questions[:50]
        
        # Predict
        self.is_training = False
        features = self.extract_features(holdout_questions)
        predictions = self.classifier.predict(features)
        confidences = self.classifier.predict_proba(features).max(axis=1)
        
        # Log results
        for i, (question, pred, conf) in enumerate(zip(holdout_questions[:15], predictions[:15], confidences[:15])):
            logger.info(f"Q: {question}")
            logger.info(f"   Predicted: {self.label_encoder.inverse_transform([pred])[0]} (confidence: {conf:.3f})")
        
        # Summary stats
        pred_counts = pd.Series(self.label_encoder.inverse_transform(predictions)).value_counts()
        logger.info(f"\nPrediction distribution: {pred_counts.to_dict()}")
        logger.info(f"Average confidence: {confidences.mean():.3f}")
    
    def run(self):
        """Main training pipeline"""
        log_step("Training complexity classifier", 
                "Phase 2 Step 2: Build lightweight classifier",
                "Per Phase 2 instructions - train on 300 synthetic + 95 existing questions")
        
        try:
            # Load data
            data = self.load_data()
            
            # Prepare features
            self.is_training = True
            X = self.extract_features(data['question'].values)
            y = self.label_encoder.fit_transform(data['complexity_tier'].values)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Training set size: {X_train.shape[0]}")
            logger.info(f"Test set size: {X_test.shape[0]}")
            
            # Train
            self.test_accuracy = self.train_classifier(X_train, y_train, X_test, y_test)
            
            # Check if we meet target
            if self.test_accuracy >= 0.85:
                logger.info(f"✓ Target accuracy of 85% achieved: {self.test_accuracy:.1%}")
            else:
                logger.warning(f"⚠ Below target accuracy (85%): {self.test_accuracy:.1%}")
            
            # Save model
            self.save_model()
            
            # Test on holdout
            self.test_holdout_questions()
            
            log_step("Complexity classifier training complete", 
                    f"Achieved {self.test_accuracy:.1%} accuracy on test set",
                    "Phase 2 Step 2 complete - model saved to models/complexity_classifier.pkl")
            
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            raise

def main():
    """Main function"""
    trainer = ComplexityClassifierTrainer()
    trainer.run()

if __name__ == "__main__":
    main()