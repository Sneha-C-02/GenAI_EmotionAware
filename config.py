"""
config.py — Central configuration for research_v2 pipeline.
All paths, hyperparameters, and constants live here.
Edit this file to switch models, datasets, or experiment settings.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# ROOT PATHS (edit these to match your Linux workstation mount points)
# ---------------------------------------------------------------------------

ROOT_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(ROOT_DIR, "data")
MODELS_DIR     = os.path.join(ROOT_DIR, "models")
RAG_DIR        = os.path.join(ROOT_DIR, "rag")
RESULTS_DIR    = os.path.join(ROOT_DIR, "results")
FIGURES_DIR    = os.path.join(ROOT_DIR, "figures")

# Dataset subdirectories
ED_DIR         = os.path.join(DATA_DIR, "empatheticdialogues")   # EmpatheticDialogues
ESCONV_DIR     = os.path.join(DATA_DIR, "esconv")                # ESConv

# Model paths — edit to your HuggingFace cache or local paths
LLM_MODEL_PATH = os.environ.get(
    "LLM_MODEL_PATH",
    "/home/user/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2"
)
GATE_MODEL_DIR       = os.path.join(MODELS_DIR, "gate")
GATE_BASE_MODEL      = "distilbert-base-uncased"   # cached locally
EMBED_MODEL_NAME     = "sentence-transformers/all-MiniLM-L6-v2"
EMOTION_MODEL_NAME   = "SamLowe/roberta-base-go_emotions"

# ---------------------------------------------------------------------------
# RAG CONFIG
# ---------------------------------------------------------------------------

@dataclass
class RAGConfig:
    embed_model_name: str  = EMBED_MODEL_NAME
    valence_weight:   float = 3.0
    arousal_weight:   float = 1.5
    dominance_weight: float = 0.5
    top_k:            int   = 3
    search_k_factor:  int   = 6
    mode: str               = "vad_augmented"   # semantic_only | vad_augmented | valence_gated
    max_corpus_words: int   = 80                # filter corpus entries longer than this
    dedup_threshold:  float = 0.85              # cosine similarity threshold for deduplication

RAG_CONFIG = RAGConfig()

# ---------------------------------------------------------------------------
# GENERATION CONFIG
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    max_new_tokens:       int   = 150
    temperature:          float = 0.7
    top_p:                float = 0.9
    repetition_penalty:   float = 1.1
    max_retries:          int   = 3
    load_in_4bit:         bool  = True
    compute_dtype:        str   = "float16"   # float16 | bfloat16

GEN_CONFIG = GenerationConfig()

# ---------------------------------------------------------------------------
# EPISODIC MEMORY CONFIG
# ---------------------------------------------------------------------------

@dataclass
class MemoryConfig:
    window:                   int   = 7
    summary_interval:         int   = 3
    escalation_arousal_slope: float = 0.15
    escalation_valence_ceil:  float = -0.35
    escalation_min_turns:     int   = 2
    arc_improving_threshold:  float = 0.10
    arc_declining_threshold:  float = -0.10

MEMORY_CONFIG = MemoryConfig()

# ---------------------------------------------------------------------------
# BEHAVIORAL GATE CONFIG
# ---------------------------------------------------------------------------

@dataclass
class GateConfig:
    base_model:     str   = GATE_BASE_MODEL
    output_dir:     str   = GATE_MODEL_DIR
    threshold:      float = 0.5
    epochs:         int   = 5
    batch_size:     int   = 16
    learning_rate:  float = 2e-5
    val_split:      float = 0.15
    max_length:     int   = 128

GATE_CONFIG = GateConfig()

# ---------------------------------------------------------------------------
# EVALUATION CONFIG
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    n_eval_items:     int   = 100              # target eval set size
    conditions:       str   = "ABCDEF"
    drift_threshold:  float = 0.5             # TDR: cosine sim below this = drift
    n_bootstrap:      int   = 1000            # bootstrap CI samples
    alpha:            float = 0.05            # significance level
    # EAS weights
    eas_w_air:        float = 0.35
    eas_w_ecs:        float = 0.25
    eas_w_tdr:        float = 0.20
    eas_w_bscore:     float = 0.15
    eas_w_fallback:   float = 0.05

EVAL_CONFIG = EvalConfig()

# ---------------------------------------------------------------------------
# OFFLINE ENFORCEMENT
# ---------------------------------------------------------------------------

def enforce_offline():
    """Call at module init to ensure no network calls are made."""
    os.environ["HF_HUB_OFFLINE"]      = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"]  = "1"

# ---------------------------------------------------------------------------
# DIRECTORY CREATION
# ---------------------------------------------------------------------------

def ensure_dirs():
    for d in [DATA_DIR, MODELS_DIR, RAG_DIR, RESULTS_DIR, FIGURES_DIR,
              ED_DIR, ESCONV_DIR, GATE_MODEL_DIR]:
        os.makedirs(d, exist_ok=True)

ensure_dirs()
