"""
corpus_builder.py — Advice-Free Behavioral Exemplar Corpus Builder
===================================================================
Builds and validates the retrieval corpus from:
  1. EmpatheticDialogues (primary) — listener turns, advice-filtered
  2. ESConv (secondary) — reflection/validation strategy turns only
  3. Seed exemplars (fallback, always available offline)

Usage:
    python corpus_builder.py build     # build corpus
    python corpus_builder.py analyze   # coverage report
    python corpus_builder.py validate  # check for advice leakage
"""

import os, re, json, csv
from typing import List, Tuple, Optional, Dict

from config import ED_DIR, ESCONV_DIR, RAG_DIR, enforce_offline

# ---------------------------------------------------------------------------
# ADVICE / VALIDATION PATTERNS
# ---------------------------------------------------------------------------

_ADVICE_PATS = [re.compile(p, re.IGNORECASE) for p in [
    r"\byou should\b", r"\btry to\b", r"\bconsider\b", r"\breach out\b",
    r"\bremember to\b", r"\bmake sure\b", r"\bif i were you\b",
    r"\bthe best thing\b", r"\bat least\b", r"\bjust relax\b",
    r"\bdon't worry\b", r"\bforgiveness\b", r"\byou need to\b",
    r"\bone thing you\b", r"\bmy advice\b", r"\bI suggest\b",
]]

_VALID_PATS = [re.compile(p, re.IGNORECASE) for p in [
    r"\bthat sounds\b", r"\bmakes sense\b", r"\bunderstandable\b",
    r"\bcompletely valid\b", r"\breally hard\b", r"\bgenuinely\b",
    r"\byour feelings\b", r"\bthat must\b", r"\bso painful\b",
    r"\bI can (imagine|understand|see)\b", r"\bthat's (really|so)\b",
]]

# ESConv strategy labels that are advice-free
ESCONV_SAFE_STRATEGIES = {
    "Reflection of feelings", "Self-disclosure",
    "Affirmation and Reassurance", "Others",
}

def _has_advice(text: str) -> bool:
    return any(p.search(text) for p in _ADVICE_PATS)

def _has_validation(text: str) -> bool:
    return any(p.search(text) for p in _VALID_PATS)

def _word_count(text: str) -> int:
    return len(text.split())

# ---------------------------------------------------------------------------
# SEED CORPUS (always available, offline)
# ---------------------------------------------------------------------------

SEED_EXEMPLARS = [
    "That sounds really painful. Having your trust broken like that is genuinely hard.",
    "It makes complete sense that you'd feel this way given what happened.",
    "What you're going through sounds incredibly isolating.",
    "That kind of betrayal cuts deep, and your feelings are completely valid.",
    "That must have stung — feeling left out by people you care about.",
    "It sounds like you're carrying a lot right now, and that's genuinely heavy.",
    "Having someone dismiss your feelings like that sounds awful.",
    "Being alone when everyone else has somewhere to go hits in a particular way.",
    "What happened sounds genuinely unfair, and it makes sense you'd feel angry.",
    "That kind of pressure all at once would overwhelm anyone.",
    "Your grief is completely real — loss is loss, regardless of what anyone says.",
    "It sounds like you've been holding a lot of this in. That's exhausting.",
    "That must feel like such a violation of your privacy and your trust.",
    "It makes sense you'd feel anxious with that much on your plate.",
    "What they said was dismissive, and it makes sense that it hurt.",
    "You're not overreacting — your feelings are proportionate to what happened.",
    "That kind of rejection from someone you were counting on is really painful.",
    "Feeling invisible to the people around you is one of the loneliest things.",
    "That must have felt like such a gut punch.",
    "Your feelings about this make complete sense.",
    "What you're describing sounds exhausting and scary at the same time.",
    "That kind of betrayal from a family member hits differently.",
    "It sounds like you needed them to show up for you, and they didn't.",
    "That's a lot to be carrying alone.",
    "Being dismissed by someone who was supposed to understand you — that hurts.",
    "Your reaction seems completely reasonable given what happened.",
    "That kind of pain doesn't just go away quickly, and that's okay.",
    "What you're feeling right now is real and it matters.",
    "That sounds like such a painful place to be in.",
    "Having so much pressure with no relief in sight is genuinely overwhelming.",
    "That loneliness — sitting alone when everyone else is somewhere — is real.",
    "It makes sense you'd feel lost with that much uncertainty.",
    "What you went through sounds genuinely unfair.",
    "That kind of dismissal on top of everything else is a lot to hold.",
    "Your feelings about what happened are completely understandable.",
]

# ---------------------------------------------------------------------------
# EMPATHETICDIALOGUES LOADER
# ---------------------------------------------------------------------------

def _load_empatheticdialogues(split: str = "train") -> List[str]:
    """Extract advice-free listener turns from EmpatheticDialogues CSV."""
    path = os.path.join(ED_DIR, f"{split}.csv")
    if not os.path.exists(path):
        print(f"[corpus] EmpatheticDialogues {split}.csv not found at {path}.")
        return []

    turns: List[str] = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                utterance = row.get("utterance", "").strip().replace("_comma_", ",")
                prompt    = row.get("prompt", "")
                # Listener turns are responses (not the opening prompt)
                if not utterance or utterance == prompt:
                    continue
                if _word_count(utterance) > 80 or _word_count(utterance) < 8:
                    continue
                if _has_advice(utterance):
                    continue
                if not _has_validation(utterance):
                    continue
                turns.append(utterance)
    except Exception as e:
        print(f"[corpus] ED load error: {e}")

    print(f"[corpus] EmpatheticDialogues: {len(turns)} advice-free listener turns")
    return turns

# ---------------------------------------------------------------------------
# ESCONV LOADER
# ---------------------------------------------------------------------------

def _load_esconv() -> List[str]:
    """Extract advice-free turns from ESConv using strategy labels."""
    turns: List[str] = []

    for fname in ("train.json", "valid.json", "test.json"):
        path = os.path.join(ESCONV_DIR, fname)
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for conv in data:
                dialog = conv.get("dialog", [])
                for turn in dialog:
                    if turn.get("speaker") != "sys":
                        continue
                    strategy = turn.get("strategy", "")
                    text     = turn.get("text", "").strip()
                    if strategy not in ESCONV_SAFE_STRATEGIES:
                        continue
                    if _word_count(text) > 80 or _word_count(text) < 8:
                        continue
                    if _has_advice(text):
                        continue
                    turns.append(text)
        except Exception as e:
            print(f"[corpus] ESConv {fname} error: {e}")

    print(f"[corpus] ESConv: {len(turns)} advice-free turns (safe strategies)")
    return turns

# ---------------------------------------------------------------------------
# DEDUPLICATION
# ---------------------------------------------------------------------------

def _deduplicate(texts: List[str], threshold: float = 0.85) -> List[str]:
    """
    Remove near-duplicate entries using sentence embedding cosine similarity.
    Falls back to exact-match dedup if sentence-transformers not available.
    """
    try:
        enforce_offline()
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from config import EMBED_MODEL_NAME

        model = SentenceTransformer(EMBED_MODEL_NAME, local_files_only=True)
        embs  = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        norms = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

        kept = []
        kept_embs = []
        for i, (text, emb) in enumerate(zip(texts, norms)):
            if not kept_embs:
                kept.append(text); kept_embs.append(emb); continue
            sims = np.array(kept_embs) @ emb
            if sims.max() < threshold:
                kept.append(text); kept_embs.append(emb)

        print(f"[corpus] Dedup: {len(texts)} → {len(kept)} (threshold={threshold})")
        return kept

    except Exception as e:
        print(f"[corpus] Embedding dedup failed ({e}), using exact-match dedup.")
        seen = set()
        deduped = []
        for t in texts:
            key = t.lower().strip()
            if key not in seen:
                seen.add(key); deduped.append(t)
        return deduped

# ---------------------------------------------------------------------------
# MAIN BUILD FUNCTION
# ---------------------------------------------------------------------------

def build_corpus(
    use_ed:    bool = True,
    use_esconv: bool = True,
    use_seed:  bool = True,
    dedup:     bool = True,
    save:      bool = True,
    vad_annotate: bool = False,
    emotion_model=None,
) -> Tuple[List[str], Optional[List[Tuple[float, float, float]]]]:
    """
    Build the advice-free behavioral exemplar corpus.

    Args:
        use_ed:         Include EmpatheticDialogues listener turns
        use_esconv:     Include ESConv reflection/validation turns
        use_seed:       Include seed exemplars (always included as fallback)
        dedup:          Run semantic deduplication
        save:           Save corpus.json to RAG_DIR
        vad_annotate:   Compute VAD for each entry (requires emotion_model)
        emotion_model:  EmotionModel instance for VAD annotation

    Returns:
        (texts, vads) — vads is None if vad_annotate=False
    """
    all_texts: List[str] = []

    if use_seed:
        all_texts.extend(SEED_EXEMPLARS)
        print(f"[corpus] Seed exemplars: {len(SEED_EXEMPLARS)}")

    if use_ed:
        all_texts.extend(_load_empatheticdialogues("train"))
        all_texts.extend(_load_empatheticdialogues("valid"))

    if use_esconv:
        all_texts.extend(_load_esconv())

    # Remove exact duplicates first (fast)
    seen = set()
    unique = []
    for t in all_texts:
        k = t.lower().strip()
        if k not in seen:
            seen.add(k); unique.append(t)
    all_texts = unique

    print(f"[corpus] After exact-dedup: {len(all_texts)} entries")

    if dedup and len(all_texts) > len(SEED_EXEMPLARS):
        all_texts = _deduplicate(all_texts)

    print(f"[corpus] Final corpus size: {len(all_texts)}")

    # VAD annotation
    vads = None
    if vad_annotate and emotion_model is not None:
        print("[corpus] Annotating corpus with VAD...")
        results = emotion_model.predict(all_texts, batch_size=32)
        vads = [(r["valence"], r["arousal"], r["dominance"]) for r in results]

    # Save
    if save:
        os.makedirs(RAG_DIR, exist_ok=True)
        corpus_path = os.path.join(RAG_DIR, "corpus.json")
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(all_texts, f, indent=2, ensure_ascii=False)
        print(f"[corpus] Saved: {corpus_path}")

        if vads is not None:
            vad_path = os.path.join(RAG_DIR, "corpus_vad.json")
            with open(vad_path, "w") as f:
                json.dump([list(v) for v in vads], f)
            print(f"[corpus] Saved: {vad_path}")

    return all_texts, vads

# ---------------------------------------------------------------------------
# ANALYSIS / VALIDATION
# ---------------------------------------------------------------------------

def analyze_corpus():
    path = os.path.join(RAG_DIR, "corpus.json")
    if not os.path.exists(path):
        print("[corpus] No corpus found. Run: python corpus_builder.py build"); return

    with open(path, encoding="utf-8") as f:
        texts = json.load(f)

    advice_count = sum(1 for t in texts if _has_advice(t))
    valid_count  = sum(1 for t in texts if _has_validation(t))
    wc           = [_word_count(t) for t in texts]

    print(f"\n[corpus] === Corpus Analysis ===")
    print(f"  Total entries:          {len(texts)}")
    print(f"  Advice contamination:   {advice_count} ({100*advice_count/len(texts):.1f}%)")
    print(f"  With validation signal: {valid_count} ({100*valid_count/len(texts):.1f}%)")
    print(f"  Word count: min={min(wc)} mean={sum(wc)/len(wc):.1f} max={max(wc)}")

def validate_corpus():
    path = os.path.join(RAG_DIR, "corpus.json")
    if not os.path.exists(path):
        print("[corpus] No corpus found."); return

    with open(path, encoding="utf-8") as f:
        texts = json.load(f)

    leaks = [(i, t) for i, t in enumerate(texts) if _has_advice(t)]
    if leaks:
        print(f"[corpus] ⚠ {len(leaks)} advice leaks found:")
        for i, t in leaks[:5]:
            print(f"  [{i}] {t[:80]}")
    else:
        print(f"[corpus] ✓ Zero advice patterns in {len(texts)} entries.")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"

    if cmd == "build":
        build_corpus(use_ed=True, use_esconv=True, use_seed=True)
    elif cmd == "analyze":
        analyze_corpus()
    elif cmd == "validate":
        validate_corpus()
    else:
        print("Usage: python corpus_builder.py [build|analyze|validate]")
