"""
metrics.py — Complete Evaluation Metrics for Emotion-Aware RAG
==============================================================
Implements all automated metrics for the ablation study:

  AIR   — Advice Injection Rate          (↓ lower = better)
  ECS   — Emotional Congruence Score     (↑ higher = better)
  TDR   — Topic Drift Rate               (↓ lower = better)
  BScore— BERTScore proxy (offline)      (↑ higher = better)
  EAS   — Emotional Alignment Score      (↑ composite, higher = better)
  RAC   — RAG Advice Contamination       (↓ retrieval quality)
  MIR   — Minimization Rate              (↓ lower = better)
  IRQ   — Interrogation Rate             (↓ lower = better)

Statistical tests (McNemar, Wilcoxon, bootstrap CI) included.
"""

import re, json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# ADVICE / MINIMIZER PATTERNS
# ---------------------------------------------------------------------------

ADVICE_PATTERNS = [
    r"\byou should\b", r"\btry to\b", r"\byou need to\b",
    r"\bit'?s (always|better|important)\b", r"\bwill help you\b",
    r"\bmake sure\b", r"\bconsider\b", r"\btake care of yourself\b",
    r"\btalking to\b", r"\breach out\b", r"\bremember to\b",
    r"\bone thing (you|that)\b", r"\bif i were you\b", r"\bmy advice\b",
    r"\bi (would |'d )suggest\b", r"\bthe best thing\b",
    r"\bwhat you (should|need to|can do)\b", r"\bat least\b",
    r"\blook on the bright side\b", r"\bjust relax\b", r"\bcalm down\b",
    r"\bdon't worry\b", r"\bit will be okay\b", r"\bforgiveness\b",
]
MINIMIZER_PATTERNS = [
    r"\bit could be worse\b", r"\bsilver lining\b", r"\bstay positive\b",
    r"\beverything happens for a reason\b", r"\btime heals\b",
    r"\byou'?ll? be fine\b", r"\bnot that bad\b",
]

_COMPILED_ADVICE    = [re.compile(p, re.IGNORECASE) for p in ADVICE_PATTERNS]
_COMPILED_MINIMIZER = [re.compile(p, re.IGNORECASE) for p in MINIMIZER_PATTERNS]

# ---------------------------------------------------------------------------
# GOLD REFERENCES (BERTScore proxy)
# ---------------------------------------------------------------------------

GOLD_REFERENCES: Dict[str, List[str]] = {
    "relationship_conflict": [
        "That sounds really painful. Having your trust broken like that is genuinely hurtful, "
        "and it makes complete sense that you'd feel hurt and angry. How are you holding up?",
        "That must have felt like such a deep violation of your trust. What happened sounds "
        "genuinely hurtful. How has it been sitting with you since?",
    ],
    "social_withdrawal": [
        "That sounds really isolating. Being the only one left behind when everyone else has "
        "somewhere to go can feel incredibly lonely. How has it been feeling?",
        "That kind of aloneness — when everyone else has somewhere to be — hits in a particular way. "
        "How are you doing with it?",
    ],
    "stress_overload": [
        "That sounds genuinely overwhelming. Having that much pressure all at once with no clear "
        "way forward is a really scary place to be. How are you feeling about it?",
        "Three finals and nowhere to start — that kind of weight is crushing. It makes complete "
        "sense you'd feel this panicked. How are you holding yourself together?",
    ],
    "interpersonal_conflict": [
        "That sounds really hurtful. Having someone dismiss your feelings like that is a form "
        "of invalidation that cuts deep. How are you feeling about what was said?",
        "Your feelings are completely real, and having them dismissed must have felt awful. "
        "That kind of invalidation on top of everything is genuinely painful. How are you doing?",
    ],
    "grief_loss": [
        "Losing someone — or something — you love is one of the most painful things. Your grief "
        "is completely real and it makes total sense you'd feel this way. How are you holding up?",
        "That kind of loss leaves a hole that doesn't just disappear. What you're feeling makes "
        "complete sense. How are you doing with it all?",
    ],
    "uncertainty": [
        "Not knowing what's going to happen, and having no control over it — that's a genuinely "
        "frightening place to be. Your anxiety about it makes total sense. How are you feeling?",
        "That kind of uncertainty is really hard to sit with. It makes sense you'd feel scared. "
        "How are you managing with all of this right now?",
    ],
}

# ---------------------------------------------------------------------------
# INDIVIDUAL METRICS
# ---------------------------------------------------------------------------

def advice_injection_rate(responses: List[str]) -> Dict:
    flags = [any(p.search(r) for p in _COMPILED_ADVICE) for r in responses]
    n = len(responses)
    return {
        "rate":         sum(flags) / n if n > 0 else 0.0,
        "count":        sum(flags),
        "total":        n,
        "per_response": flags,
    }

def minimizer_rate(responses: List[str]) -> Dict:
    flags = [any(p.search(r) for p in _COMPILED_MINIMIZER) for r in responses]
    n = len(responses)
    return {
        "rate":         sum(flags) / n if n > 0 else 0.0,
        "per_response": flags,
    }

def interrogation_rate(responses: List[str]) -> Dict:
    flags = [r.count("?") > 2 for r in responses]
    return {"rate": sum(flags) / len(responses) if responses else 0.0, "per_response": flags}

def emotional_congruence_score(
    input_vads:    List[Tuple[float, float, float]],
    response_vads: List[Tuple[float, float, float]],
) -> Dict:
    assert len(input_vads) == len(response_vads)
    scores, valence_aligned = [], []
    for iv, rv in zip(input_vads, response_vads):
        iv_a, rv_a = np.array(iv, dtype=float), np.array(rv, dtype=float)
        denom      = (np.linalg.norm(iv_a) * np.linalg.norm(rv_a)) + 1e-8
        scores.append(float(np.dot(iv_a, rv_a) / denom))
        valence_aligned.append((1 if iv[0] >= 0 else -1) == (1 if rv[0] >= 0 else -1))
    return {
        "mean_ecs":              float(np.mean(scores)),
        "std_ecs":               float(np.std(scores)),
        "valence_alignment_rate": float(np.mean(valence_aligned)),
        "per_sample":            scores,
    }

def compute_response_vads(responses: List[str], emotion_model) -> List[Tuple[float, float, float]]:
    results = emotion_model.predict(responses)
    return [(r["valence"], r["arousal"], r["dominance"]) for r in results]

def topic_drift_rate(
    input_texts:    List[str],
    response_texts: List[str],
    encoder,
    drift_threshold: float = 0.5,
) -> Dict:
    i_embs = encoder.encode(input_texts,    convert_to_numpy=True, show_progress_bar=False)
    r_embs = encoder.encode(response_texts, convert_to_numpy=True, show_progress_bar=False)
    i_embs /= (np.linalg.norm(i_embs, axis=1, keepdims=True) + 1e-8)
    r_embs /= (np.linalg.norm(r_embs, axis=1, keepdims=True) + 1e-8)
    sims   = (i_embs * r_embs).sum(axis=1).tolist()
    flags  = [s < drift_threshold for s in sims]
    return {
        "mean_similarity": float(np.mean(sims)),
        "drift_rate":      float(np.mean(flags)),
        "per_sample":      sims,
    }

def bertscore_proxy(responses: List[str], scenario_labels: List[str], encoder) -> Dict:
    scores = []
    for resp, label in zip(responses, scenario_labels):
        refs = GOLD_REFERENCES.get(label, [])
        if not refs:
            scores.append(0.0); continue
        r_emb  = encoder.encode([resp], convert_to_numpy=True)[0]
        rf_emb = encoder.encode(refs,   convert_to_numpy=True)
        r_emb  /= (np.linalg.norm(r_emb) + 1e-8)
        rf_emb /= (np.linalg.norm(rf_emb, axis=1, keepdims=True) + 1e-8)
        scores.append(float((rf_emb @ r_emb).max()))
    return {"mean_score": float(np.mean(scores)) if scores else 0.0, "per_sample": scores}

def rag_advice_contamination(retrieved_docs_per_query: List[List[str]]) -> Dict:
    per_query = []
    for docs in retrieved_docs_per_query:
        if not docs: per_query.append(0.0); continue
        contaminated = sum(any(p.search(d) for p in _COMPILED_ADVICE) for d in docs)
        per_query.append(contaminated / len(docs))
    return {
        "mean_contamination_rate": float(np.mean(per_query)) if per_query else 0.0,
        "per_query":               per_query,
    }

def retry_economics(retry_logs: List[Dict]) -> Dict:
    if not retry_logs: return {}
    attempts  = [r.get("attempts", 1) for r in retry_logs]
    fallbacks = [r.get("used_fallback", False) for r in retry_logs]
    latencies = [r["latency_ms"] for r in retry_logs if "latency_ms" in r]
    n = len(retry_logs)
    return {
        "mean_attempts":     float(np.mean(attempts)),
        "fallback_rate":     float(np.mean(fallbacks)),
        "single_pass_rate":  float(sum(a == 1 for a in attempts) / n),
        "mean_latency_ms":   float(np.mean(latencies)) if latencies else None,
    }

# ---------------------------------------------------------------------------
# EAS — Emotional Alignment Score (composite)
# ---------------------------------------------------------------------------

def emotional_alignment_score(
    air:          float,
    ecs:          float,
    tdr:          float,
    bscore:       float,
    fallback_rate: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    EAS = weighted composite of component metrics.
    Default weights: AIR=0.35, ECS=0.25, TDR=0.20, BScore=0.15, Fallback=0.05

    All components mapped to [0, 1] where higher = better.
    """
    w = weights or {
        "air": 0.35, "ecs": 0.25, "tdr": 0.20, "bscore": 0.15, "fallback": 0.05
    }
    return (
        w["air"]      * (1.0 - air)           +
        w["ecs"]      * max(0.0, (ecs + 1) / 2)  +  # map [-1,1] → [0,1]
        w["tdr"]      * (1.0 - tdr)           +
        w["bscore"]   * bscore                +
        w["fallback"] * (1.0 - fallback_rate)
    )

# ---------------------------------------------------------------------------
# STATISTICAL TESTS
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: List[float],
    n_boot: int   = 1000,
    alpha:  float = 0.05,
) -> Tuple[float, float, float]:
    """Returns (mean, lower_ci, upper_ci)."""
    arr        = np.array(values)
    boot_means = [np.mean(np.random.choice(arr, len(arr))) for _ in range(n_boot)]
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(arr)), float(lo), float(hi)

def mcnemar_test(flags_a: List[bool], flags_b: List[bool]) -> Dict:
    """McNemar test for comparing two binary classifiers (e.g., AIR condition A vs B)."""
    assert len(flags_a) == len(flags_b)
    b = sum(a and not b for a, b in zip(flags_a, flags_b))   # A=1, B=0
    c = sum(not a and b for a, b in zip(flags_a, flags_b))   # A=0, B=1
    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p    = scipy_stats.chi2.sf(stat, df=1)
    return {"statistic": float(stat), "p_value": float(p), "b": b, "c": c}

def wilcoxon_test(scores_a: List[float], scores_b: List[float]) -> Dict:
    """Paired Wilcoxon signed-rank test for continuous metrics (ECS, BScore)."""
    try:
        stat, p = scipy_stats.wilcoxon(scores_a, scores_b)
        d       = (np.mean(scores_b) - np.mean(scores_a)) / (np.std(scores_a) + 1e-8)
        return {"statistic": float(stat), "p_value": float(p), "cohens_d": float(d)}
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------------------------
# INTER-ANNOTATOR AGREEMENT
# ---------------------------------------------------------------------------

def cohens_kappa(ann1: List[int], ann2: List[int]) -> float:
    assert len(ann1) == len(ann2)
    n     = len(ann1)
    p_o   = sum(a == b for a, b in zip(ann1, ann2)) / n
    cats  = sorted(set(ann1) | set(ann2))
    p_e   = sum((ann1.count(c) / n) * (ann2.count(c) / n) for c in cats)
    denom = 1 - p_e
    return (p_o - p_e) / denom if abs(denom) > 1e-9 else 1.0

ANNOTATION_RUBRIC = {
    "emotional_validation":    "Names and validates the specific emotion (1–5)",
    "situational_specificity": "References the specific situation, not generic distress (1–5)",
    "advice_absence":          "Free of unsolicited advice (1=full of advice, 5=none) (1–5)",
    "naturalness":             "Sounds like a caring human, not a chatbot (1–5)",
    "closing_question":        "Ends with open, gentle, user-feeling-focused question (1–5)",
}

# ---------------------------------------------------------------------------
# METRICS REPORT
# ---------------------------------------------------------------------------

@dataclass
class MetricsReport:
    condition_name: str
    n_samples:      int
    air:            Optional[Dict] = None
    minimizer:      Optional[Dict] = None
    interrogation:  Optional[Dict] = None
    ecs:            Optional[Dict] = None
    tdr:            Optional[Dict] = None
    bertscore:      Optional[Dict] = None
    rag_contamination: Optional[Dict] = None
    retry:          Optional[Dict] = None
    eas:            Optional[float] = None
    human_scores:   Optional[Dict] = None

    def compute_eas(self) -> float:
        air_r = self.air["rate"]           if self.air       else 0.5
        ecs_r = self.ecs["mean_ecs"]       if self.ecs       else 0.0
        tdr_r = self.tdr["drift_rate"]     if self.tdr       else 0.5
        bs_r  = self.bertscore["mean_score"] if self.bertscore else 0.0
        fb_r  = self.retry["fallback_rate"]  if self.retry     else 0.0
        self.eas = emotional_alignment_score(air_r, ecs_r, tdr_r, bs_r, fb_r)
        return self.eas

    def to_dict(self) -> Dict:
        return {
            "condition":        self.condition_name,
            "n":                self.n_samples,
            "AIR":              self.air["rate"]            if self.air       else None,
            "ECS":              self.ecs["mean_ecs"]        if self.ecs       else None,
            "VAL_ALIGN":        self.ecs["valence_alignment_rate"] if self.ecs else None,
            "TDR":              self.tdr["drift_rate"]      if self.tdr       else None,
            "BERTScore":        self.bertscore["mean_score"] if self.bertscore else None,
            "RAG_contamination":self.rag_contamination["mean_contamination_rate"]
                                if self.rag_contamination else None,
            "fallback_rate":    self.retry["fallback_rate"] if self.retry     else None,
            "EAS":              self.eas,
        }

    def print_summary(self) -> None:
        print(f"\n{'='*60}\n  {self.condition_name}  (n={self.n_samples})\n{'='*60}")
        if self.air:
            print(f"  AIR            : {self.air['rate']:.3f}  ({self.air['count']}/{self.air['total']})")
        if self.minimizer:
            print(f"  Minimizer Rate : {self.minimizer['rate']:.3f}")
        if self.ecs:
            print(f"  ECS            : {self.ecs['mean_ecs']:.3f} ± {self.ecs['std_ecs']:.3f}")
            print(f"  Valence Align  : {self.ecs['valence_alignment_rate']:.3f}")
        if self.tdr:
            print(f"  TDR            : {self.tdr['drift_rate']:.3f}")
        if self.bertscore:
            print(f"  BERTScore      : {self.bertscore['mean_score']:.3f}")
        if self.rag_contamination:
            print(f"  RAG Contamination: {self.rag_contamination['mean_contamination_rate']:.3f}")
        if self.retry:
            print(f"  Fallback Rate  : {self.retry['fallback_rate']:.3f}")
        if self.eas is not None:
            print(f"  EAS (composite): {self.eas:.3f}")

    @staticmethod
    def to_latex(reports: List["MetricsReport"], output_path: str) -> None:
        cols = ["condition", "n", "AIR", "ECS", "TDR", "BERTScore", "EAS", "fallback_rate"]
        hdrs = {
            "condition":    "Condition",
            "n":            "$N$",
            "AIR":          r"AIR~$\downarrow$",
            "ECS":          r"ECS~$\uparrow$",
            "TDR":          r"TDR~$\downarrow$",
            "BERTScore":    r"BScore~$\uparrow$",
            "EAS":          r"EAS~$\uparrow$",
            "fallback_rate":r"Fallback~$\downarrow$",
        }
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else ("—" if v is None else str(v))
        rows = [r.to_dict() for r in reports]

        lines = [
            r"\begin{table}[h]", r"\centering", r"\small",
            r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}",
            r"\toprule",
            " & ".join(hdrs[c] for c in cols) + r" \\",
            r"\midrule",
        ]
        for row in rows:
            lines.append(" & ".join(fmt(row.get(c)) for c in cols) + r" \\")
        lines += [
            r"\bottomrule", r"\end{tabular}",
            r"\caption{Ablation results across six conditions. "
            r"AIR=Advice Injection Rate, ECS=Emotional Congruence Score, "
            r"TDR=Topic Drift Rate, EAS=Emotional Alignment Score (composite).}",
            r"\label{tab:ablation}", r"\end{table}",
        ]
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        print(f"[metrics] LaTeX table saved → {output_path}")


# ---------------------------------------------------------------------------
# CONVENIENCE FUNCTION
# ---------------------------------------------------------------------------

def compute_all(
    inputs:         List[str],
    responses:      List[str],
    input_vads:     List[Tuple[float, float, float]],
    scenario_labels: List[str],
    condition_name: str,
    emotion_model=None,
    encoder=None,
    retrieved_docs_per_query: Optional[List[List[str]]] = None,
    retry_logs:     Optional[List[Dict]] = None,
) -> MetricsReport:
    n      = len(inputs)
    report = MetricsReport(condition_name=condition_name, n_samples=n)

    report.air          = advice_injection_rate(responses)
    report.minimizer    = minimizer_rate(responses)
    report.interrogation= interrogation_rate(responses)

    if emotion_model is not None:
        resp_vads   = compute_response_vads(responses, emotion_model)
        report.ecs  = emotional_congruence_score(input_vads, resp_vads)

    if encoder is not None:
        report.tdr       = topic_drift_rate(inputs, responses, encoder)
        report.bertscore = bertscore_proxy(responses, scenario_labels, encoder)

    if retrieved_docs_per_query is not None:
        report.rag_contamination = rag_advice_contamination(retrieved_docs_per_query)

    if retry_logs is not None:
        report.retry = retry_economics(retry_logs)

    report.compute_eas()
    return report
