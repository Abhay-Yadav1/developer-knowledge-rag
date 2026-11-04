# RAG Improvement Log (Hinglish)

Yeh file har improvement ka audit log hai: kya badla, kyun badla, aur uska practical effect kya hai.

## 2026-11-02 — Retrieval quality: lightweight lexical reranking

### Kya badla

- `src/retrieval/retriever.py` ab vector store se requested `k` ke bajay `4 × k` candidates leta hai.
- Candidate ordering ko query ke exact words, code identifiers (jaise `MAX_RETRIES`), aur file path ke matching terms se refine karta hai.
- Final output phir bhi zyada se zyada `k` chunks rakhta hai.
- Original cosine similarity score ko nahi badla; `min_score` ab bhi usi score par apply hota hai.

### Kyun badla

Semantic embeddings meaning ke liye achhe hote hain, lekin exact function names, config keys, ya error identifiers par kabhi kabhi kamzor ho sakte hain. Wider candidate pool aur halka lexical rerank aise exact matches ko top par la sakta hai, bina nayi runtime dependency ya model ke.

### Kaise kaam karta hai

Query aur chunk text ko lowercase word tokens mein todte hain; underscore wale identifiers ek token rehte hain. Query ke kitne tokens chunk mein milte hain, is coverage se candidate ko chhota ranking bonus milta hai. Path match ko aur chhota bonus milta hai. Cosine score unchanged hai aur threshold pass karne ke liye zaroori hai.

### Is change ki seema

Yeh reranker sirf vector search ke laaye hue candidate pool ko reorder kar sakta hai. Jo relevant chunk pool mein aaya hi nahi, use yeh recover nahi karega. Isliye yeh full hybrid vector + keyword search nahi hai. Agle retrieval step mein proper keyword retrieval, jaise BM25, add karke same evaluation set par compare karna chahiye.

### Verification

Is change ke baad tests run nahi kiye gaye. Evaluation ke liye index aur question set same rakhkar baseline aur naye run ke `hit rate`, `MRR`, `precision@k`, aur per-question changes compare karein.
