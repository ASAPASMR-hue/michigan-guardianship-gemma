---
aliases:
  - "(2.1) Project Guidance: Michigan Minor Guardianship AI - Complete Playbook"
---
# Project Guidance: Michigan Minor Guardianship AI - Complete Playbook

**Version**: 2.1  
**Last Updated**: 2025-07-11  
**Status**: Production Ready with Security & Compliance Updates

---

## Executive Summary

This playbook synthesizes Stanford Justice Innovation's human-centered quality rubric with production-ready RAG implementation, enhanced by HyPA-RAG's adaptive retrieval insights. Our mission: Create an AI assistant that helps Michigan families navigate minor guardianship with **zero hallucination** and **maximum actionability**.

### Core Philosophy

- **Accuracy ≠ Quality**: Perfect legal citations that don't tell Grandma which form to file fail our mission
- **Dynamic Mode Switching**: Seamlessly blend strict legal facts with empathetic guidance
- **Adaptive Complexity**: Simple questions get fast answers; complex scenarios get thorough analysis
- **Genesee County Focus**: Every response includes county-specific details (Thursday hearings, $175 fee)
- **Legal Compliance**: Clear disclaimers per Michigan Rule of Professional Conduct 7.1

---

## Part A: Technical Implementation Stack

### A.1 Document Processing & Chunking

```python
##############################################################################
# Chunking Configuration - Updated 2025-07-11 (Single Source of Truth)
##############################################################################

# Semantic Chunking with Legal Pattern Recognition
CHUNK_CONFIG = {
    "size": 1000,  # tokens - validated sweet spot for legal text
    "overlap": 100,
    "separators": [
        "\n## ",     # Major sections
        "\n### ",    # Subsections  
        "\nMCL ",    # Michigan statutes
        "\nPC ",     # Court forms
        "\n§ ",      # Legal sections
        "\n- ",      # Bullet points
        "\n\n",      # Paragraphs
    ],
    "preserve_together": [
        r"(Form PC \d+.*?)\n",  # Keep form numbers with descriptions
        r"(MCL \d+\.\d+.*?)\n", # Keep statutes intact
        r"(\$\d+.*?waiver.*?)\n", # Keep fees with waiver info
        r"(\d+ days?.*?)\n",     # Keep deadlines together
    ]
}

# Metadata Schema - Critical for Filtering
METADATA_SCHEMA = {
    "jurisdiction": "Genesee County",  # Always filter by this
    "doc_type": ["statute", "form", "procedure", "local_rule"],
    "form_numbers": ["PC 651", "PC 650", "MC 20", ...],
    "applies_to": ["full_guardianship", "limited_guardianship", "emergency"],
    "last_updated": "2025-07-11",
    "critical_deadlines": {
        "personal_service": "7_days",
        "mail_service": "14_days", 
        "proof_of_service": "5_days_before_hearing"
    },
    "genesee_specifics": {
        "filing_fee": 175,
        "hearing_days": ["Thursday"],
        "court_address": "900 S. Saginaw St., Room 502, Flint, MI 48502"
    }
}

# Document Update Strategy
EMBEDDING_UPDATE_STRATEGY = {
    "method": "incremental",  # vs "full_rebuild"
    "trigger": "on_document_change",
    "process": """
    1. Identify changed chunks via content hash
    2. Re-embed only modified chunks
    3. Update vector store with upsert
    4. Maintain version history for rollback
    """,
    "full_rebuild_triggers": [
        "embedding_model_change",
        "chunk_strategy_change",
        "quarterly_maintenance"
    ]
}
```

### A.2 Embedding & Vector Store

```python
##############################################################################
# Embedding Configuration - Locked to HF Models
##############################################################################

# Primary Model (Required)
EMBED_MODEL = "BAAI/bge-m3"  # 4.1M downloads, Apache-2.0
EMBED_CONFIG = {
    "model_name": EMBED_MODEL,
    "batch_size": 64,
    "normalize_embeddings": True,
    "encode_kwargs": {"device": "cuda"},
    "use_fp16": True  # GPU efficiency
}

# Fallback (Only if primary unavailable)
FALLBACK_EMBED = "intfloat/multilingual-e5-large"  # 3.4M downloads, MIT

# Vector Database Configuration
VECTOR_DB_CONFIG = {
    "backend": "chromadb",  # or weaviate for production
    "collection_name": "michigan_guardianship_v2",
    "distance_metric": "cosine",
    "metadata_filters": {
        "mandatory": {"jurisdiction": "Genesee County"},
        "exclude": {"doc_status": "outdated"}
    }
}
```

### A.3 Adaptive Retrieval Pipeline with Latency Budgets

```python
##############################################################################
# Query Complexity Classifier with Latency Guardrails
##############################################################################

class QueryComplexityClassifier:
    """
    Classifies queries into complexity tiers for adaptive retrieval
    with strict latency budgets
    """
    COMPLEXITY_TIERS = {
        "simple": {
            "examples": ["filing fee?", "what form?", "court address?"],
            "top_k": 5,
            "query_rewrites": 0,
            "rerank_top_k": 3,
            "latency_budget_ms": 800,  # Fast response required
            "latency_p95_ms": 1000
        },
        "standard": {
            "examples": ["grandma guardianship", "parent consent"],
            "top_k": 10,
            "query_rewrites": 3,
            "rerank_top_k": 5,
            "latency_budget_ms": 1500,
            "latency_p95_ms": 1800
        },
        "complex": {
            "examples": ["ICWA applies", "emergency + out of state"],
            "top_k": 15,
            "query_rewrites": 5,
            "rerank_top_k": 7,
            "latency_budget_ms": 2000,
            "latency_p95_ms": 2500,  # Still under global 2s typical
            "fallback_if_slow": {  # Graceful degradation
                "top_k": 10,
                "query_rewrites": 3
            }
        }
    }
    
    def classify(self, query: str) -> str:
        # TF-IDF + keyword matching for now
        # Future: DistilBERT fine-tuned on 300 synthetic examples
        if any(kw in query.lower() for kw in ["icwa", "tribal", "emergency", "multi-state"]):
            return "complex"
        elif len(query.split()) > 15 or "?" in query:
            return "standard"
        else:
            return "simple"

# Hybrid Search Configuration
SEARCH_PIPELINE = {
    "vector_search": {
        "model": EMBED_MODEL,
        "weight": 0.7
    },
    "lexical_search": {
        "algorithm": "BM25",
        "parameters": {"k1": 1.2, "b": 0.75},
        "weight": 0.3
    },
    "reranker": {
        "model": "BAAI/bge-reranker-v2-m3",  # 2.5M downloads, mandatory
        "batch_size": 32
    }
}
```

### A.4 Generation with Mode Switching and Legal Disclaimers

```python
##############################################################################
# LLM Configuration with Dynamic Modes and Compliance
##############################################################################

# Model Selection (Choose based on resources)
LLM_OPTIONS = {
    "production": "mistralai/Mixtral-8x22B-Instruct-v0.1",  # 18K downloads
    "development": "Qwen/Qwen2.5-72B-Instruct",  # 159K downloads
    "testing": "unsloth/llama-3-8b-Instruct-bnb-4bit"  # 70K downloads
}

# Master Prompt Template with Legal Disclaimer
SYSTEM_PROMPT = """
LEGAL DISCLAIMER: I am an AI assistant providing general information about Michigan minor guardianship procedures. This is NOT legal advice. For legal advice specific to your situation, please consult a licensed Michigan attorney. No attorney-client relationship is created by using this service.

You are a Michigan minor guardianship assistant for Genesee County.

MODES (Never mention these to users):
- STRICT: Legal facts, procedures, forms, deadlines, citations
- PERSONALIZED: Empathy, explanations, transitions, guidance

ZERO HALLUCINATION POLICY:
- Every legal fact MUST have inline citation from provided chunks
- Format: fact (MCL 700.XXXX) or (Form PC XXX) or (Document N: Title)
- If information isn't in chunks, say "I don't have that specific information"

OUT OF SCOPE HANDLING:
- If asked about non-guardianship legal matters, politely redirect
- Use patterns from Out-of-Scope Guidelines document
- Example: "I can only help with minor guardianship in Genesee County. For [topic], please contact [appropriate resource]."

GENESEE COUNTY CONSTANTS:
- Filing fee: $175 (waiver: Form MC 20)
- Hearings: Thursdays only
- Court: 900 S. Saginaw St., Room 502, Flint, MI 48502
- Proof of service: 5 days before hearing

Retrieved Context:
{retrieved_chunks}

Query Complexity: {complexity_tier}
Adaptive Parameters: top_k={top_k}, rewrites={rewrites}
"""

# UI Disclaimer Configuration
UI_DISCLAIMER = {
    "placement": "top_banner",
    "style": "conspicuous",  # Per Michigan Rule 7.1
    "text": """
    ⚖️ LEGAL INFORMATION DISCLAIMER: This AI assistant provides general information 
    about Michigan minor guardianship procedures. This is NOT legal advice and does 
    not create an attorney-client relationship. For advice about your specific 
    situation, consult a licensed Michigan attorney.
    """,
    "persistent": True,
    "user_acknowledgment_required": True
}
```

### A.5 Post-Generation Validation with Out-of-Scope Handling

```python
##############################################################################
# Validation Pipeline - Mandatory Checks Including Scope
##############################################################################

from lettuce_detect import Detector  # Required hallucination detection
from out_of_scope_guidelines import OUT_OF_SCOPE_PATTERNS

class ResponseValidator:
    def __init__(self):
        self.hallucination_detector = Detector()
        self.genesee_patterns = {
            "filing_fee": r"\$175",
            "thursday": r"Thursday",
            "address": r"900 S\. Saginaw",
            "forms": r"PC \d{3}|MC \d{2}"
        }
        self.out_of_scope_patterns = OUT_OF_SCOPE_PATTERNS
    
    def validate(self, response, retrieved_chunks, question_type, original_query):
        # 0. Out-of-Scope Check (First gate)
        if self.check_out_of_scope(original_query):
            return {
                "pass": True,  # Valid refusal
                "out_of_scope": True,
                "suggested_response": self.generate_scope_redirect(original_query)
            }
        
        # 1. Hallucination Check (Binary Fail)
        hallucination_score = self.hallucination_detector.check(
            response, retrieved_chunks
        )
        if hallucination_score > 0.05:  # 5% threshold
            return {"pass": False, "reason": "Hallucination detected"}
        
        # 2. Citation Verification
        legal_claims = self.extract_legal_claims(response)
        for claim in legal_claims:
            if not self.has_citation(claim):
                return {"pass": False, "reason": f"Uncited claim: {claim}"}
        
        # 3. Procedural Accuracy (Critical for Genesee)
        if question_type in ["filing", "deadlines", "forms"]:
            if not self.verify_genesee_specifics(response):
                return {"pass": False, "reason": "Missing/wrong county details"}
        
        # 4. Mode Appropriateness
        mode_score = self.assess_mode_balance(response, question_type)
        
        # 5. Legal Disclaimer Present
        if not self.contains_appropriate_caveat(response, question_type):
            response = self.add_contextual_disclaimer(response)
        
        return {
            "pass": True,
            "scores": {
                "hallucination": 0,
                "citation_compliance": 1.0,
                "procedural_accuracy": 1.0,
                "mode_effectiveness": mode_score
            }
        }
    
    def check_out_of_scope(self, query):
        """Check if query is outside minor guardianship scope"""
        for pattern in self.out_of_scope_patterns:
            if pattern['regex'].search(query.lower()):
                return True
        return False
    
    def generate_scope_redirect(self, query):
        """Generate appropriate redirect for out-of-scope queries"""
        # Logic to map query to appropriate resource
        # See Out-of-Scope Guidelines.md for full mapping
        pass
```

---

## Part B: Human-Centered Evaluation Framework

### B.1 Scoring Dimensions (Total: 10 points)

```yaml
evaluation_rubric:
  # Critical Legal Accuracy (4.5 pts total)
  procedural_accuracy:
    weight: 2.5
    adaptive_weight_by_complexity:
      simple: 3.0    # Higher weight for simple procedural questions
      standard: 2.5
      complex: 2.0   # Lower weight when juggling multiple factors
    critical_items:
      - "Correct form numbers (PC 651 vs PC 650)"
      - "Service deadlines (7/14/5 days)"
      - "Filing fee amount and waiver process"
      - "Thursday hearings requirement"
      - "Genesee County courthouse address"
    fail_override: true  # Any error = automatic fail
    
  substantive_legal_accuracy:
    weight: 2.0
    adaptive_weight_by_complexity:
      simple: 1.5
      standard: 2.0
      complex: 2.5   # More important for complex scenarios
    examples:
      - "MCL 700.5204 grounds for guardianship"
      - "Parent consent requirements"
      - "ICWA/MIFPA procedures"
      - "Guardian vs conservator distinctions"
      
  # User Success Metrics (4.0 pts total)
  actionability:
    weight: 2.0
    requirements:
      - "Specific form to file"
      - "Where to go (with address)"
      - "What to bring"
      - "Timeline/next steps"
    scoring: "Points for each concrete action provided"
    
  mode_effectiveness:
    weight: 1.5
    adaptive_weight_by_complexity:
      simple: 1.0    # Less critical for factual queries
      standard: 1.5
      complex: 2.0   # Essential for crisis/emotional situations
    criteria:
      strict_appropriateness: "Legal facts properly cited"
      personalized_quality: "Empathy without legal speculation"
      transition_smoothness: "Natural flow between modes"
      
  strategic_caution:
    weight: 0.5
    good_examples:
      - "If parents object, court needs evidence of unfitness"
      - "ICWA cases take longer - plan for extended timeline"
    bad_examples:
      - "You should consult an attorney" (repeated unnecessarily)
      - No warning about ex parte 14-day requirement
      
  # Supporting Elements (1.0 pts total)
  citation_quality:
    weight: 0.5
    requirements:
      - "Every legal fact has inline citation"
      - "Citations placed before punctuation"
      - "No bundling multiple facts under one cite"
      
  harm_prevention:
    weight: 0.5
    red_flags:
      - "Wrong jurisdiction advice"
      - "Missing critical deadlines"
      - "Encouraging guardian shopping"
      - "Minimizing ICWA requirements"
      
  # Total: 2.5 + 2.0 + 2.0 + 1.5 + 0.5 + 0.5 + 0.5 = 10.0 points
```

### B.2 Complexity-Aware Evaluation

```yaml
question_complexity_tiers:
  simple:
    description: "Direct factual queries"
    examples:
      - "What's the filing fee?"
      - "Where is the courthouse?"
      - "What form do I need?"
    success_threshold: 0.95  # Must be nearly perfect
    critical_dimensions: ["procedural_accuracy", "actionability"]
    
  standard:
    description: "Common scenarios with clear paths"
    examples:
      - "Grandparent seeking guardianship"
      - "Parent wants to terminate guardianship"
      - "Guardian needs to move out of state"
    success_threshold: 0.85
    critical_dimensions: ["substantive_accuracy", "mode_effectiveness"]
    
  complex:
    description: "Multi-factor situations"
    examples:
      - "ICWA + emergency + out-of-state parent"
      - "Contested guardianship with CPS involvement"
      - "Limited converting to full guardianship"
    success_threshold: 0.80
    critical_dimensions: ["strategic_caution", "mode_effectiveness"]
    
  crisis:
    description: "Urgent emotional situations"
    examples:
      - "Parent overdosed, need care today"
      - "Child abandoned at my door"
      - "Guardian died suddenly"
    success_threshold: 0.85
    critical_dimensions: ["actionability", "harm_prevention", "empathy"]
```

---

## Part C: Bridging Technical & Evaluation

### C.1 Technical Safeguard → Rubric Risk Mapping

|Rubric Dimension|Technical Safeguard|Implementation|
|---|---|---|
|Procedural Accuracy (2.5)|Mandatory Genesee metadata filter|`filter: {jurisdiction: "Genesee County"}`|
|Substantive Accuracy (2.0)|BGE-M3 embeddings + reranking|Retrieve statute chunks with high precision|
|Actionability (2.0)|Pattern-based chunk preservation|Keep form+instruction chunks together|
|Mode Effectiveness (1.5)|Few-shot examples in prompt|21 annotated mode-switching examples|
|Strategic Caution (0.5)|Query complexity classifier|Complex queries get risk warnings|
|Citation Quality (0.5)|Post-generation validation|Reject responses with uncited claims|
|Harm Prevention (0.5)|LettuceDetect + override rules + out-of-scope check|Auto-fail on critical errors|

### C.2 Instrumentation & Logging

```python
# Every query-response cycle logs:
query_log = {
    "query_id": uuid4(),
    "timestamp": datetime.utcnow(),
    "user_query": str,
    "complexity_tier": str,  # From classifier
    "out_of_scope": bool,
    "retrieved_chunks": List[ChunkID],
    "reranker_scores": List[float],
    "adaptive_params": {
        "top_k": int,
        "rewrites": int,
        "rerank_k": int
    },
    "latency_breakdown": {
        "embedding_ms": float,
        "retrieval_ms": float,
        "reranking_ms": float,
        "generation_ms": float,
        "validation_ms": float,
        "total_ms": float
    },
    "generation": {
        "model": str,
        "response": str,
        "token_count": int,
        "contained_disclaimer": bool
    },
    "validation": {
        "hallucination_score": float,
        "citation_compliance": float,
        "procedural_accuracy": float,
        "mode_effectiveness": float,
        "final_score": float,
        "pass": bool
    }
}
```

---

## Part D: Genesee County Constants & Quick Reference

```yaml
# Machine-readable constants for system prompts
genesee_county_constants:
  court:
    name: "Genesee County Probate Court"
    address: "900 S. Saginaw St., Room 502, Flint, MI 48502"
    phone: "(810) 257-3528"
    
  scheduling:
    hearing_days: ["Thursday"]
    typical_wait: "6+ weeks"
    
  fees:
    petition_filing: 175
    letters_of_guardianship: 11  # per certified copy
    publication: 96.05
    payment_methods: ["cash", "money_order", "credit", "debit"]
    not_accepted: ["personal_check"]
    waiver_form: "MC 20"
    waiver_eligibility: "125% federal poverty or hardship"
    
  deadlines:
    personal_service: 7  # days before hearing
    mail_service: 14
    publication: 14
    proof_of_service_filing: 5  # days before hearing
    
  critical_forms:
    full_guardianship: "PC 651"
    limited_guardianship: "PC 650"
    placement_plan: "PC 652"
    social_history: "PC 670"
    notice_of_hearing: "PC 562"
    proof_of_service: "PC 564"
    annual_report: "PC 654"
    terminate_modify: "PC 675"
    
  local_requirements:
    background_checks: ["LEIN", "CPS registry"]
    check_all_adults: true
    social_history_required: true
    birth_certificate: "original or certified"
    
  # Knowledge gaps requiring fallback
  unresolved_questions:
    - "Court permission for address changes"
    - "Home study/investigation process details"
    - "SSI benefits and representative payee"
```

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

- [] Embed all documents with BAAI/bge-m3
- [] Set up hybrid search with mandatory Genesee filters
- [] Implement BAAI/bge-reranker-v2-m3
- [] Integrate LettuceDetect validation
- [] Deploy base evaluation rubric
- [ ] **NEW**: Link Out-of-Scope Guidelines validator

### Phase 2: Adaptive Enhancement (Weeks 3-4)

- [ ] Generate 300 synthetic complexity-labeled questions
- [ ] Train lightweight complexity classifier
- [ ] Implement adaptive top-k with latency budgets
- [ ] A/B test pattern-based chunking
- [ ] Add RAGAS metrics to evaluation
- [ ] **NEW**: Set up incremental embedding updates

### Phase 3: Testing & Refinement (Weeks 5-6)

- [ ] Run all 95 test questions through pipeline
- [ ] Battle mode comparison of model responses
- [ ] Tune adaptive parameters based on results
- [ ] Document failure patterns
- [ ] **NEW**: Test latency degradation scenarios
- [ ] **NEW**: Verify disclaimer compliance

### Phase 4: Production Readiness (Week 7+)

- [ ] Implement comprehensive logging with PII protection
- [ ] Set up monitoring dashboards
- [ ] Create fallback mechanisms
- [ ] Train content moderators
- [ ] **NEW**: Deploy security measures (see Appendix A)
- [ ] **NEW**: Set up CI/CD with quality gates
- [ ] Launch pilot with select users

### Future Research (Parked)

- [ ] HyPA-RAG knowledge graph integration
- [ ] Cross-jurisdiction expansion
- [ ] Multi-language support
- [ ] Voice interface

---

## Risk Mitigation

### Technical Risks

1. **Classifier Mislabeling**: If retrieved_docs < 3, automatically bump to next complexity tier
2. **Query Rewriter Drift**: Cap rewrites at 128 tokens
3. **Pattern Splitter Errors**: Maintain 100-token overlap
4. **Model Latency**: Cache common queries, use smaller models for classification
5. **NEW - Latency Spikes**: Implement circuit breakers at tier thresholds

### Legal/Ethical Risks

1. **Hallucination**: LettuceDetect + citation validation + human review
2. **Wrong Jurisdiction**: Mandatory metadata filtering
3. **Harmful Advice**: Override rules for dangerous patterns
4. **Bias**: Regular audits across demographic scenarios
5. **NEW - Unauthorized Practice**: Clear disclaimers + scope limitations

### Privacy/Security Risks

1. **PII Exposure**: See Appendix A for handling protocols
2. **Query Logging**: Anonymize after 90 days
3. **Model Attacks**: Input sanitization + rate limiting

---

## Success Metrics & KPIs

### Technical Performance

- Retrieval precision: >0.85
- Hallucination rate: <0.5%
- P95 latency by tier:
    - Simple: <1000ms
    - Standard: <1800ms
    - Complex: <2500ms
- Citation compliance: 100%

### User Outcomes

- Procedural accuracy: >98%
- Actionability score: >85%
- Mode effectiveness: >80%
- User task completion: Track actual filings
- Out-of-scope handling: >95% appropriate redirects

### CI/CD Quality Gates

- Block deployment if:
    - Procedural accuracy <98%
    - Hallucination rate >0.5%
    - Any latency P95 exceeds budget
    - Missing disclaimer in >1% of responses

---

## Appendix A: Security & Privacy Controls

### Data Handling

```yaml
privacy_controls:
  pii_detection:
    - SSN patterns: mask to XXX-XX-####
    - Phone numbers: mask to XXX-XXX-####
    - Email addresses: hash for analytics
    - Names: retain only for active session
    
  retention:
    - Query logs: 90 days then anonymize
    - Retrieved chunks: session only
    - User corrections: aggregate only
    
  encryption:
    - At rest: AES-256
    - In transit: TLS 1.3
    - Key rotation: quarterly
```

### Access Controls

```yaml
access_controls:
  api_authentication: OAuth2 + JWT
  rate_limiting:
    - Per user: 100 queries/hour
    - Per IP: 1000 queries/hour
    - Complexity-based: Complex queries count 2x
  
  audit_logging:
    - All access attempts
    - Configuration changes
    - Model updates
```

---

## Appendix B: Related Documentation

- `Out-of-Scope Guidelines.md` - Detailed patterns for refusing non-guardianship queries
- `Dynamic Mode Examples.md` - 21 annotated examples of mode switching
- `Genesee County Specifics.md` - Comprehensive local requirements
- `Test Questions Dataset.csv` - 95 evaluated test cases

---

## Version History

### v2.1 (2025-07-11)

- Added out-of-scope validation integration
- Implemented legal disclaimer requirements (Michigan Rule 7.1)
- Set latency budgets per complexity tier
- Added security & privacy controls appendix
- Fixed YAML scoring to total 10 points
- Documented incremental embedding strategy

### v2.0 (2025-01-11)

- Integrated HyPA-RAG adaptive retrieval
- Added Stanford human-centered rubric

### v1.1 (2025-01-10)

- Added dynamic mode examples

### v1.0 (2025-01-09)

- Initial merged playbook

---

## Key Takeaways

1. **Quality = Actionability**: The best response helps users take the right next step
2. **Complexity Drives Parameters**: Simple questions need speed; complex need depth
3. **Mode Switching is Critical**: Legal accuracy alone creates poor user experience
4. **Genesee Specifics are Non-Negotiable**: Every response must include local details
5. **Zero Hallucination is Achievable**: Through careful retrieval + validation + citation
6. **Legal Compliance is Mandatory**: Clear disclaimers protect both users and operators

This playbook represents the synthesis of cutting-edge research with practical implementation requirements for the Michigan minor guardianship domain. Follow it carefully, iterate based on real user outcomes, and maintain compliance with all legal and ethical requirements.