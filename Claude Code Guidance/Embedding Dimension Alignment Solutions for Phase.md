<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

## Embedding Dimension Alignment Solutions for Phase 2

Based on your dimension mismatch challenge and the comprehensive research findings, here are my thoughts on the optimal path forward for your Michigan Guardianship RAG system:

### **Primary Recommendation: Switch to all-mpnet-base-v2**

The **most straightforward and effective solution** is to replace all-MiniLM-L6-v2 with **all-mpnet-base-v2** for local query embeddings. This model outputs **768 dimensions**, providing perfect alignment with your Google text-embedding-004 document embeddings.[^1][^2]

**Key advantages:**

- **Perfect dimensional compatibility** - eliminates the core alignment problem
- **Superior semantic quality** - consistently outperforms smaller models in legal/domain-specific tasks[^3][^4]
- **Proven track record** - extensively validated in systematic review and legal document applications[^5][^6]
- **Maintains privacy benefits** - fully local inference with no API calls
- **Simple implementation** - direct drop-in replacement requiring minimal code changes


### **Gemma 3 Models: Promising but Unverified**

While **Gemma 3 models show impressive capabilities**, they present several challenges for your specific use case:[^7][^8][^9]

**Potential benefits:**

- **State-of-the-art performance** - latest Google technology with strong benchmarks[^10][^11]
- **Multiple size options** - from 270M to 27B parameters allowing resource optimization
- **Legal domain potential** - could be fine-tuned for guardianship-specific terminology

**Critical unknowns:**

- **Embedding dimensions are unclear** - the models are primarily designed for text generation, not embedding extraction
- **Extraction methodology uncertain** - would require research into how to properly extract meaningful embeddings
- **Resource requirements** - even the smallest viable model (4B) requires 6.4GB memory[^8]

**Recommendation**: Consider Gemma 3 for **future research** rather than immediate Phase 2 implementation, particularly once clearer documentation on embedding extraction becomes available.

### **Alternative Solutions Analysis**

**Linear Projection Layer (Advanced Option):**
If you want to stick with all-MiniLM-L6-v2, training a **384→768 projection layer** could work well. This approach:[^12][^13]

- Trains a small neural network to map 384-dim vectors to 768-dim space
- Can be optimized on your specific legal corpus for domain adaptation
- Maintains privacy benefits while potentially improving legal-specific performance

**Avoid These Approaches:**

- **Simple truncation** of 768-dim vectors to 384 - causes significant semantic information loss[^14][^15]
- **Basic PCA projection** without validation - high risk in legal precision tasks[^16][^17]
- **Complex dual-encoder systems** - unnecessary complexity for your current needs[^18][^19]


### **Implementation Strategy for Phase 2**

**Immediate Path (Recommended):**

```python
# In local_embedder.py
from sentence_transformers import SentenceTransformer

# Replace all-MiniLM-L6-v2 with all-mpnet-base-v2
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
query_embedding = model.encode("How do I file for guardianship in Genesee County?")
# Output: 768-dimensional vector, perfectly aligned with Pinecone
```

**Performance optimization recommendations:**

1. **Ensure L2 normalization** for cosine similarity consistency[^20][^14]
2. **Implement hybrid search** combining dense vectors with BM25 for 15-20% precision improvement[^21][^22]
3. **Add metadata filtering** to leverage your rich legal-specific metadata (MCL/MCR statutes, ICWA flags)[^23]
4. **Consider reranking** with your existing LLM for enhanced precision[^19]

### **Future Research Direction**

For Phase 3 or future iterations, **investigate Gemma 3's embedding capabilities** once:

- Clear documentation on embedding extraction methods emerges
- Community validation of dimension outputs occurs
- Your system is stable with all-mpnet-base-v2 as baseline

The research shows that **smaller, well-aligned models often outperform larger ones** in RAG systems when combined with proper reranking, suggesting your privacy-first, locally-optimized approach is architecturally sound.[^19]

**Bottom line**: The all-mpnet-base-v2 solution provides immediate, proven compatibility while maintaining your system's privacy advantages and avoiding the risks associated with experimental approaches for a sensitive legal application.
<span style="display:none">[^24][^25][^26][^27][^28][^29][^30][^31][^32][^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54][^55][^56][^57][^58][^59][^60][^61][^62][^63][^64][^65][^66][^67][^68][^69][^70][^71][^72][^73][^74][^75][^76][^77]</span>

<div style="text-align: center">⁂</div>

[^1]: https://dataloop.ai/library/model/sentence-transformers_all-mpnet-base-v2/

[^2]: https://huggingface.co/sentence-transformers/all-mpnet-base-v2

[^3]: https://www.mdpi.com/2078-2489/15/2/68

[^4]: https://www.mdpi.com/2504-2289/9/3/67

[^5]: https://www.polipapers.upv.es/index.php/WPOM/article/view/22282

[^6]: https://journals.orclever.com/oprd/article/view/516

[^7]: https://huggingface.co/blog/gemma3

[^8]: https://ai.google.dev/gemma/docs/core

[^9]: https://ollama.com/library/gemma3

[^10]: https://www.mdpi.com/2504-2289/9/5/141

[^11]: https://ijsrem.com/download/fine-tuning-small-llms-for-high-quality-semantic-search-a-cost-efficient-alternative-to-foundation-models/

[^12]: https://trilogi.ac.id/journal/ks/index.php/JISA/article/view/2179

[^13]: https://arxiv.org/abs/2502.02603

[^14]: https://osanseviero.github.io/hackerllama/blog/posts/sentence_embeddings/

[^15]: https://www.reddit.com/r/MachineLearning/comments/1b2yc4f/d_rag_dimensionality_reduction_for_embeddings/

[^16]: https://arxiv.org/html/2505.00105v1

[^17]: https://arxiv.org/html/2504.08386v1

[^18]: https://dl.acm.org/doi/10.1145/3543873.3587668

[^19]: https://arxiv.org/html/2506.00049v1

[^20]: https://ai.google.dev/gemini-api/docs/embeddings

[^21]: https://haystack.deepset.ai/blog/hybrid-retrieval

[^22]: https://qdrant.tech/articles/hybrid-search/

[^23]: grok_report.pdf

[^24]: https://arxiv.org/abs/2504.18562

[^25]: http://medrxiv.org/lookup/doi/10.1101/2024.04.17.24305983

[^26]: https://arxiv.org/abs/2506.13467

[^27]: https://arxiv.org/abs/2505.18441

[^28]: https://aclanthology.org/2021.emnlp-main.372

[^29]: https://www.jmir.org/2025/1/e66344

[^30]: https://arxiv.org/abs/2507.16847

[^31]: https://www.semanticscholar.org/paper/17f6ea8abc26b5b05d5ada95bb3fd9db08eecddc

[^32]: https://arxiv.org/pdf/2403.08295.pdf

[^33]: https://arxiv.org/html/2404.07839

[^34]: https://arxiv.org/pdf/2409.10173.pdf

[^35]: https://arxiv.org/pdf/2503.02142.pdf

[^36]: http://arxiv.org/pdf/2503.20278.pdf

[^37]: https://arxiv.org/html/2503.19786

[^38]: https://www.aclweb.org/anthology/D19-1369.pdf

[^39]: https://arxiv.org/pdf/1909.01761.pdf

[^40]: https://arxiv.org/pdf/2404.01331.pdf

[^41]: https://arxiv.org/pdf/2101.07577.pdf

[^42]: https://developers.googleblog.com/en/introducing-gemma-3-270m/

[^43]: https://blog.google/technology/developers/gemma-3/

[^44]: https://ai.google.dev/gemma/docs/gemma-3n

[^45]: https://huggingface.co/google/gemma-3-1b-it

[^46]: https://zilliz.com/ai-faq/what-is-the-typical-dimensionality-of-sentence-embeddings-produced-by-sentence-transformer-models

[^47]: https://github.com/open-webui/open-webui/issues/11279

[^48]: https://cloud.google.com/vertex-ai/generative-ai/docs/open-models/use-gemma

[^49]: https://milvus.io/ai-quick-reference/what-is-the-typical-dimensionality-of-sentence-embeddings-produced-by-sentence-transformer-models

[^50]: https://zilliz.com/ai-faq/why-do-i-see-a-dimension-mismatch-or-shape-error-when-using-embeddings-from-a-sentence-transformer-in-another-tool-or-network

[^51]: https://huggingface.co/google/gemma-3-12b-it

[^52]: https://huggingface.co/sentence-transformers/stsb-bert-base

[^53]: https://milvus.io/ai-quick-reference/why-do-i-see-a-dimension-mismatch-or-shape-error-when-using-embeddings-from-a-sentence-transformer-in-another-tool-or-network

[^54]: https://linkinghub.elsevier.com/retrieve/pii/S1877050925012384

[^55]: https://www.semanticscholar.org/paper/13362c331c84db687e6d05b5dd535162680e04ce

[^56]: https://ieeexplore.ieee.org/document/10804807/

[^57]: https://everant.org/index.php/etj/article/view/1648

[^58]: https://www.semanticscholar.org/paper/a60399550a4874d4aef13aefc9dd0a3c5c7b636c

[^59]: https://journal.ppmi.web.id/index.php/jrsit/article/view/1419

[^60]: http://arxiv.org/pdf/2410.23510.pdf

[^61]: http://arxiv.org/pdf/2310.15285.pdf

[^62]: http://arxiv.org/pdf/2402.14776.pdf

[^63]: https://www.aclweb.org/anthology/2020.emnlp-main.18.pdf

[^64]: https://arxiv.org/pdf/2408.08073.pdf

[^65]: https://arxiv.org/pdf/2108.08877.pdf

[^66]: http://arxiv.org/pdf/1907.04307.pdf

[^67]: https://arxiv.org/html/2409.11316

[^68]: http://arxiv.org/pdf/2106.03717.pdf

[^69]: https://www.aclweb.org/anthology/2020.acl-main.214.pdf

[^70]: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html

[^71]: https://docs.pinecone.io/models/all-mpnet-base-v2

[^72]: https://www.gabormelli.com/RKB/all-mpnet-base-v2

[^73]: https://docs.voxel51.com/tutorials/dimension_reduction.html

[^74]: https://collabnix.com/ollama-embedded-models-the-complete-technical-guide-to-local-ai-embeddings-in-2025/

[^75]: https://courses.grainger.illinois.edu/cs441/sp2023/lectures/Lecture 22 - Dimensionality Reduction.pdf

[^76]: https://www.mongodb.com/company/blog/technical/matryoshka-embeddings-smarter-embeddings-with-voyage-ai

[^77]: https://en.wikipedia.org/wiki/Dimensionality_reduction

