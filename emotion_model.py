"""
emotion_model.py — GoEmotions + VAD Projection + State Override (Layer 1)
=========================================================================
Usage:
    model = EmotionModel()
    result = model.predict("I feel so alone.")[0]
    # result = {"emotions": [...], "valence": -0.7, "arousal": 0.3, ...}
"""

import numpy as np
from typing import List, Dict, Union
from config import EMOTION_MODEL_NAME, enforce_offline

enforce_offline()

# ---------------------------------------------------------------------------
# GoEmotions → VAD mapping (Russell circumplex, manually assigned)
# ---------------------------------------------------------------------------

VAD_MAP = {
    "admiration":     ( 0.7,  0.4,  0.5),
    "amusement":      ( 0.8,  0.5,  0.3),
    "anger":          (-0.6,  0.8, -0.3),
    "annoyance":      (-0.4,  0.5, -0.2),
    "approval":       ( 0.5,  0.3,  0.4),
    "caring":         ( 0.6,  0.3,  0.4),
    "confusion":      (-0.2,  0.4, -0.4),
    "curiosity":      ( 0.3,  0.5,  0.2),
    "desire":         ( 0.4,  0.6,  0.3),
    "disappointment": (-0.6,  0.3, -0.4),
    "disapproval":    (-0.5,  0.4, -0.3),
    "disgust":        (-0.7,  0.6, -0.2),
    "embarrassment":  (-0.5,  0.5, -0.5),
    "excitement":     ( 0.8,  0.8,  0.4),
    "fear":           (-0.7,  0.8, -0.6),
    "gratitude":      ( 0.8,  0.3,  0.3),
    "grief":          (-0.8,  0.4, -0.5),
    "joy":            ( 0.9,  0.6,  0.5),
    "love":           ( 0.9,  0.5,  0.4),
    "nervousness":    (-0.4,  0.7, -0.5),
    "optimism":       ( 0.6,  0.4,  0.4),
    "pride":          ( 0.7,  0.5,  0.6),
    "realization":    ( 0.1,  0.4,  0.2),
    "relief":         ( 0.6,  0.2,  0.4),
    "remorse":        (-0.6,  0.4, -0.4),
    "sadness":        (-0.7,  0.3, -0.5),
    "surprise":       ( 0.2,  0.7,  0.1),
    "neutral":        ( 0.0,  0.0,  0.0),
}

# ---------------------------------------------------------------------------
# RULE-BASED STATE OVERRIDE
# ---------------------------------------------------------------------------

_OVERRIDE_RULES = [
    {
        "keywords": ["diary", "secret", "betrayed", "trust", "behind my back",
                     "cheated", "backstab", "told everyone"],
        "state":    "relationship_conflict",
    },
    {
        "keywords": ["alone", "lonely", "left out", "nobody invited", "without me",
                     "no friends", "sitting alone", "excluded"],
        "state":    "social_withdrawal",
    },
    {
        "keywords": ["finals", "exam", "deadline", "overwhelmed", "too much",
                     "falling apart", "can't breathe", "behind on everything"],
        "state":    "stress_overload",
    },
    {
        "keywords": ["overreacting", "too sensitive", "not a big deal", "get over it",
                     "dismissed", "laughed at", "made fun of", "invalidated"],
        "state":    "interpersonal_conflict",
    },
    {
        "keywords": ["died", "passed away", "lost my", "funeral", "grief",
                     "gone forever", "never coming back"],
        "state":    "grief_loss",
    },
    {
        "keywords": ["scared", "terrified", "don't know what", "uncertain",
                     "second-guessing", "no control", "afraid"],
        "state":    "uncertainty",
    },
]

def _apply_override(text: str, model_states: List[str]) -> Dict:
    text_lower = text.lower()
    for rule in _OVERRIDE_RULES:
        if any(kw in text_lower for kw in rule["keywords"]):
            return {
                "high_level_states": [rule["state"]],
                "override_active":   True,
                "override_reason":   f"Keyword match for {rule['state']}",
            }
    return {
        "high_level_states": model_states if model_states else ["unknown"],
        "override_active":   False,
        "override_reason":   None,
    }

# Emotion label → high-level state
_LABEL_TO_STATE = {
    "anger":          "interpersonal_conflict",
    "annoyance":      "interpersonal_conflict",
    "disapproval":    "interpersonal_conflict",
    "disgust":        "interpersonal_conflict",
    "sadness":        "social_withdrawal",
    "grief":          "grief_loss",
    "fear":           "uncertainty",
    "nervousness":    "uncertainty",
    "confusion":      "uncertainty",
    "disappointment": "relationship_conflict",
    "embarrassment":  "social_withdrawal",
    "remorse":        "relationship_conflict",
    "joy":            "positive_engagement",
    "love":           "positive_engagement",
    "optimism":       "positive_engagement",
    "excitement":     "positive_engagement",
    "gratitude":      "positive_engagement",
    "admiration":     "positive_engagement",
    "amusement":      "positive_engagement",
    "pride":          "positive_engagement",
    "relief":         "positive_engagement",
    "caring":         "positive_engagement",
    "curiosity":      "positive_engagement",
    "surprise":       "uncertainty",
    "desire":         "positive_engagement",
    "realization":    "uncertainty",
    "approval":       "positive_engagement",
    "neutral":        "unknown",
}


# ---------------------------------------------------------------------------
# EMOTION MODEL
# ---------------------------------------------------------------------------

class EmotionModel:

    def __init__(self, model_name: str = EMOTION_MODEL_NAME):
        enforce_offline()
        from transformers import pipeline
        self._pipe = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            device=-1,
            model_kwargs={"local_files_only": True},
        )

    def predict(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
    ) -> List[Dict]:
        if isinstance(texts, str):
            texts = [texts]

        results = []
        for i in range(0, len(texts), batch_size):
            batch   = texts[i:i + batch_size]
            outputs = self._pipe(batch)

            for text, preds in zip(batch, outputs):
                # preds is list of {"label": ..., "score": ...}
                preds_sorted = sorted(preds, key=lambda x: x["score"], reverse=True)

                # Top emotions above threshold
                top_emotions = [
                    {"label": p["label"], "score": float(p["score"])}
                    for p in preds_sorted if p["score"] > 0.1
                ][:5]

                # VAD projection (probability-weighted average)
                total_p = 0.0
                vad     = np.array([0.0, 0.0, 0.0])
                for p in preds:
                    if p["label"] in VAD_MAP:
                        score   = float(p["score"])
                        total_p += score
                        vad     += score * np.array(VAD_MAP[p["label"]])
                if total_p > 0:
                    vad /= total_p

                # High-level state from top emotion
                top_label   = preds_sorted[0]["label"] if preds_sorted else "neutral"
                model_state = _LABEL_TO_STATE.get(top_label, "unknown")

                # Override
                interpretation = _apply_override(text, [model_state])

                results.append({
                    "top_emotions":   top_emotions,
                    "valence":        float(vad[0]),
                    "arousal":        float(vad[1]),
                    "dominance":      float(vad[2]),
                    "interpretation": interpretation,
                })
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = EmotionModel()
    tests = [
        "My sister read my diary and told our parents everything.",
        "Everyone went home and I'm sitting alone.",
        "I have three finals tomorrow and I haven't started.",
        "My boyfriend said I'm overreacting about my dog dying.",
        "I'm terrified of what's going to happen.",
    ]
    for text in tests:
        r = model.predict(text)[0]
        state = r["interpretation"]["high_level_states"][0]
        ovr   = "⚡OVERRIDE" if r["interpretation"]["override_active"] else ""
        print(f"\n{text}")
        print(f"  State: {state} {ovr}")
        print(f"  VAD:   V={r['valence']:.2f} A={r['arousal']:.2f} D={r['dominance']:.2f}")
        print(f"  Top:   {', '.join(e['label'] for e in r['top_emotions'][:3])}")
