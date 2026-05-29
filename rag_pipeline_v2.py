"""
rag_pipeline_v2.py — VAD-Augmented Emotion-Aware Retrieval
===========================================================
Three ablatable retrieval modes:
  semantic_only   — baseline (Condition C in ablation)
  vad_augmented   — concatenated VAD vector (Condition D)
  valence_gated   — post-filter: only same valence-sign docs (Condition E)

Index format:
  Semantic: FAISS IndexFlatIP over 384-dim normalized text embeddings
  VAD:      FAISS IndexFlatIP over [384-dim text | V*wv | A*wa | D*wd]

Usage:
    python rag_pipeline_v2.py build    # build FAISS indices
    python rag_pipeline_v2.py test     # run retrieval comparison
    python rag_pipeline_v2.py sweep    # VAD weight sensitivity (Experiment 4)
"""

import os, json
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from config import RAGConfig, RAG_CONFIG, RAG_DIR, enforce_offline

enforce_offline()


def _index_path(name: str) -> str:
    return os.path.join(RAG_DIR, f"faiss_{name}.bin")

def _docs_path() -> str:
    return os.path.join(RAG_DIR, "corpus.json")

def _vad_path() -> str:
    return os.path.join(RAG_DIR, "corpus_vad.json")


# ---------------------------------------------------------------------------
# CORE CLASS
# ---------------------------------------------------------------------------

class EmotionAwareRAGv2:

    def __init__(self, config: RAGConfig = RAG_CONFIG):
        self.config = config
        enforce_offline()

        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(config.embed_model_name, local_files_only=True)
        self.embed_dim  = self.encoder.get_sentence_embedding_dimension()
        self.total_dim  = self.embed_dim + 3

        self.index_sem: Optional[faiss.Index] = None
        self.index_vad: Optional[faiss.Index] = None
        self.documents:   List[str]                       = []
        self.corpus_vads: List[Tuple[float, float, float]] = []

        self._load()

    # ------------------------------------------------------------------
    # BUILD
    # ------------------------------------------------------------------

    def build_index(
        self,
        corpus_texts: List[str],
        corpus_vads:  Optional[List[Tuple[float, float, float]]] = None,
        emotion_model=None,
    ) -> None:
        n = len(corpus_texts)
        print(f"[RAG] Building index for {n} documents...")

        # Text embeddings
        text_embs = self.encoder.encode(
            corpus_texts, convert_to_numpy=True,
            show_progress_bar=True, batch_size=64
        ).astype(np.float32)
        norms          = np.linalg.norm(text_embs, axis=1, keepdims=True) + 1e-8
        text_embs_norm = text_embs / norms

        # VAD annotation
        if corpus_vads is None:
            if emotion_model is None:
                raise ValueError("Provide corpus_vads or emotion_model.")
            print("[RAG] Annotating corpus VAD...")
            results     = emotion_model.predict(corpus_texts, batch_size=32)
            corpus_vads = [(r["valence"], r["arousal"], r["dominance"]) for r in results]

        vad_arr     = np.array(corpus_vads, dtype=np.float32)
        vad_weighted = vad_arr * np.array([
            self.config.valence_weight,
            self.config.arousal_weight,
            self.config.dominance_weight,
        ], dtype=np.float32)

        # Semantic-only index
        idx_sem = faiss.IndexFlatIP(self.embed_dim)
        idx_sem.add(text_embs_norm)

        # VAD-augmented index
        combined      = np.hstack((text_embs_norm, vad_weighted))
        combined_norm = combined / (np.linalg.norm(combined, axis=1, keepdims=True) + 1e-8)
        idx_vad       = faiss.IndexFlatIP(self.total_dim)
        idx_vad.add(combined_norm)

        os.makedirs(RAG_DIR, exist_ok=True)
        faiss.write_index(idx_sem, _index_path("semantic"))
        faiss.write_index(idx_vad, _index_path("vad"))
        with open(_docs_path(), "w", encoding="utf-8") as f:
            json.dump(corpus_texts, f, indent=2, ensure_ascii=False)
        with open(_vad_path(), "w") as f:
            json.dump([list(v) for v in corpus_vads], f)

        self.index_sem   = idx_sem
        self.index_vad   = idx_vad
        self.documents   = corpus_texts
        self.corpus_vads = corpus_vads
        print(f"[RAG] ✓ Index built: {n} documents")

    def _load(self) -> None:
        if os.path.exists(_index_path("semantic")):
            self.index_sem = faiss.read_index(_index_path("semantic"))
            print(f"[RAG] Loaded semantic index ({self.index_sem.ntotal} docs)")
        else:
            self.index_sem = faiss.IndexFlatIP(self.embed_dim)

        if os.path.exists(_index_path("vad")):
            self.index_vad = faiss.read_index(_index_path("vad"))
        else:
            self.index_vad = faiss.IndexFlatIP(self.total_dim)

        if os.path.exists(_docs_path()):
            with open(_docs_path(), encoding="utf-8") as f:
                self.documents = json.load(f)

        if os.path.exists(_vad_path()):
            with open(_vad_path()) as f:
                self.corpus_vads = [tuple(v) for v in json.load(f)]

    # ------------------------------------------------------------------
    # RETRIEVE
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_text:     str,
        user_state_vad: Tuple[float, float, float],
        top_k:          Optional[int] = None,
        mode:           Optional[str] = None,
        trajectory_signal: Optional[Dict] = None,  # from episodic_memory
    ) -> Tuple[List[str], Dict]:
        """
        Retrieve emotionally relevant behavioral exemplars.

        Args:
            query_text:        Raw user input (NO label injection)
            user_state_vad:    (V, A, D) from emotion_model
            top_k:             Override config top_k
            mode:              Override config mode
            trajectory_signal: Optional dict from EpisodicMemory.get_retrieval_reranking_signal()
                               Used for trajectory-aware reranking (Layer 3 upgrade)
        Returns:
            (documents, explainability_dict)
        """
        top_k  = top_k or self.config.top_k
        mode   = mode  or self.config.mode
        search_k = top_k * self.config.search_k_factor

        if not self.documents:
            return [], {"error": "empty_index", "mode": mode}

        # Encode query text only (no label injection)
        q_emb      = self.encoder.encode([query_text], convert_to_numpy=True).astype(np.float32)[0]
        q_emb_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)

        # Build query vector
        if mode == "semantic_only":
            q_vec = q_emb_norm.reshape(1, -1)
            index = self.index_sem

        elif mode in ("vad_augmented", "valence_gated"):
            v, a, d = user_state_vad
            q_vad = np.array([
                v * self.config.valence_weight,
                a * self.config.arousal_weight,
                d * self.config.dominance_weight,
            ], dtype=np.float32)
            q_combined = np.concatenate([q_emb_norm, q_vad])
            q_combined /= (np.linalg.norm(q_combined) + 1e-8)
            q_vec = q_combined.reshape(1, -1)
            index = self.index_vad
        else:
            raise ValueError(f"Unknown mode: {mode}")

        distances, indices = index.search(q_vec, min(search_k, len(self.documents)))

        candidates = []
        for score, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.documents):
                doc_vad = self.corpus_vads[idx] if self.corpus_vads else (0., 0., 0.)
                candidates.append({
                    "text": self.documents[idx],
                    "score": float(score),
                    "idx": int(idx),
                    "doc_vad": doc_vad,
                })

        filtered_out = 0
        # Valence gate post-filter
        if mode == "valence_gated" and self.corpus_vads:
            q_vsign  = 1 if user_state_vad[0] >= 0 else -1
            pre      = len(candidates)
            filtered = [c for c in candidates
                        if (1 if c["doc_vad"][0] >= 0 else -1) == q_vsign]
            candidates   = filtered if filtered else candidates  # fallback
            filtered_out = pre - len(candidates)

        # Trajectory-aware reranking (if episodic memory signal available)
        if trajectory_signal and trajectory_signal.get("arc_direction") == "declining":
            # Prefer lower-arousal, slightly-more-positive docs for declining arcs
            candidates.sort(key=lambda c: (
                c["score"] * 0.7 +
                max(0, c["doc_vad"][0]) * 0.2 +
                (1 - abs(c["doc_vad"][1])) * 0.1
            ), reverse=True)
        else:
            candidates.sort(key=lambda c: c["score"], reverse=True)

        final      = candidates[:top_k]
        final_docs = [c["text"] for c in final]

        reasoning = {
            "mode":               mode,
            "query_vad":          list(user_state_vad),
            "vad_weights":        [self.config.valence_weight,
                                   self.config.arousal_weight,
                                   self.config.dominance_weight],
            "n_candidates":       len(candidates) + filtered_out,
            "n_valence_filtered": filtered_out,
            "top_k_scores":       [c["score"] for c in final],
            "top_k_doc_vads":     [list(c["doc_vad"]) for c in final],
            "trajectory_signal":  trajectory_signal,
        }
        return final_docs, reasoning

    # ------------------------------------------------------------------
    # RETRIEVAL EVALUATION (Experiment 1)
    # ------------------------------------------------------------------

    def evaluate_retrieval(
        self,
        eval_queries: List[Dict],
        emotion_model,
        modes: Optional[List[str]] = None,
    ) -> Dict:
        from metrics import rag_advice_contamination
        modes = modes or ["semantic_only", "vad_augmented", "valence_gated"]
        results = {}

        for mode in modes:
            print(f"\n[RAG] Evaluating mode: {mode}")
            all_docs, valence_aligns, mean_scores = [], [], []

            for item in eval_queries:
                er  = emotion_model.predict(item["text"])[0]
                vad = (er["valence"], er["arousal"], er["dominance"])
                docs, reasoning = self.retrieve(item["text"], vad, mode=mode)
                all_docs.append(docs)
                mean_scores.append(float(np.mean(reasoning["top_k_scores"])) if reasoning["top_k_scores"] else 0.)

                q_vsign    = 1 if vad[0] >= 0 else -1
                doc_vsigns = [1 if dv[0] >= 0 else -1 for dv in reasoning["top_k_doc_vads"] if dv]
                if doc_vsigns:
                    valence_aligns.append(sum(s == q_vsign for s in doc_vsigns) / len(doc_vsigns))

            contamination = rag_advice_contamination(all_docs)
            results[mode] = {
                "advice_contamination_rate": contamination["mean_contamination_rate"],
                "valence_alignment_rate":    float(np.mean(valence_aligns)) if valence_aligns else None,
                "mean_retrieval_score":      float(np.mean(mean_scores)),
                "n_queries":                 len(eval_queries),
            }
            print(f"  Contamination: {results[mode]['advice_contamination_rate']:.3f}")
            print(f"  Valence Align: {results[mode]['valence_alignment_rate']}")

        return results

    # ------------------------------------------------------------------
    # VAD WEIGHT SWEEP (Experiment 4)
    # ------------------------------------------------------------------

    def sweep_vad_weights(
        self,
        eval_queries: List[Dict],
        emotion_model,
        weight_values: Optional[List[float]] = None,
    ) -> Dict:
        weight_values  = weight_values or [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
        orig           = self.config.valence_weight
        sweep_results  = {}

        for w in weight_values:
            self.config.valence_weight = w
            res = self.evaluate_retrieval(eval_queries, emotion_model, ["vad_augmented"])
            sweep_results[str(w)] = res["vad_augmented"]

        self.config.valence_weight = orig
        return sweep_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from config import EMBED_MODEL_NAME

    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"

    if cmd == "build":
        from corpus_builder import build_corpus
        from emotion_model import EmotionModel
        emo   = EmotionModel()
        texts, vads = build_corpus(vad_annotate=True, emotion_model=emo)
        rag   = EmotionAwareRAGv2()
        rag.build_index(texts, corpus_vads=vads)

    elif cmd == "test":
        from emotion_model import EmotionModel
        rag = EmotionAwareRAGv2()
        emo = EmotionModel()
        test_inputs = [
            ("My sister read my diary and told our parents everything.", (-0.6, 0.7, -0.4)),
            ("Everyone went home and I'm sitting alone in my dorm room.", (-0.7, 0.2, -0.5)),
            ("I have three finals tomorrow and haven't started.", (-0.5, 0.8, -0.3)),
        ]
        for text, vad in test_inputs:
            print(f"\nQuery: {text[:60]}")
            for m in ["semantic_only", "vad_augmented", "valence_gated"]:
                docs, r = rag.retrieve(text, vad, mode=m)
                print(f"  [{m}] top: {docs[0][:70]}..." if docs else f"  [{m}] No docs")

    elif cmd == "sweep":
        from emotion_model import EmotionModel
        from evaluate_research import EVAL_SET
        rag   = EmotionAwareRAGv2()
        emo   = EmotionModel()
        sweep = rag.sweep_vad_weights(EVAL_SET[:20], emo)
        print(f"\n{'Weight':<8} {'Contamination':>15} {'Valence Align':>15}")
        for w, res in sweep.items():
            va = f"{res['valence_alignment_rate']:.3f}" if res.get('valence_alignment_rate') else "N/A"
            print(f"  {w:<8} {res['advice_contamination_rate']:>15.3f} {va:>15}")
