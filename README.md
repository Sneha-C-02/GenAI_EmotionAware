# Emotion-Aware RAG — Research Pipeline v2

This directory (`research_v2/`) contains the clean, publication-ready implementation of the **Emotion-Aware RAG** framework. It has been re-architected to support the six-condition ablation study for an ACL/EMNLP submission.

## Architecture Highlights
- **Layer 1: Emotion Detection** (`emotion_model.py`) — GoEmotions with VAD projection and rule-based overrides.
- **Layer 2: Episodic Memory** (`episodic_memory.py`) — Tracks 7-turn emotional trajectory, escalation detection, and provides NL context injection.
- **Layer 3: RAG** (`rag_pipeline_v2.py`) — VAD-augmented FAISS retrieval preventing semantic-only advice leakage.
- **Layer 4-5: Generation** (`evaluate_research.py`) — 4-bit local Mistral-7B inference.
- **Layer 6: Behavioral Gate** (`behavioral_gate.py`) — Trained DistilBERT classifier replacing regex filters.
- **Evaluation** (`metrics.py`) — AIR, ECS, TDR, BScore, and EAS composite.

## Setup Instructions

### 1. Download Datasets
You need to download the datasets into `data/` before building the corpus or training the gate.
*   **EmpatheticDialogues**: Place `train.csv` and `valid.csv` in `data/empatheticdialogues/`
*   **ESConv**: Place `train.json` in `data/esconv/`

### 2. Execution Order

Run the following scripts sequentially to reproduce the research findings:

#### Phase 1: Build Knowledge Base
```bash
python corpus_builder.py build
python rag_pipeline_v2.py build
```

#### Phase 2: Train Behavioral Gate
```bash
python behavioral_gate.py train --ed data/empatheticdialogues/train.csv --esconv data/esconv/train.json
```

#### Phase 3: Run the Main Ablation Study
This will run the 100-item evaluation set across all 6 conditions.
```bash
python evaluate_research.py --model /path/to/mistral --conditions ABCDEF --output results/
```
*Outputs: `ablation_metrics.json` and a LaTeX table `ablation_table.tex`*

#### Phase 4: Multi-Turn Benchmark
Validates the episodic memory module across scripted emotional arcs.
```bash
python multiturn_benchmark.py --model /path/to/mistral --output results/
```

### 3. Model Export
When ready to deploy the demo, export the trained behavioral gate to `.pth` and `.ptl`:
```bash
python behavioral_gate.py export
```

## Important Notes
*   **Offline Mode**: All scripts enforce `HF_HUB_OFFLINE=1` internally. Make sure your models (Mistral, GoEmotions, sentence-transformers) are cached locally.
*   **CUDA Memory**: The evaluation script uses `BitsAndBytesConfig` (4-bit NF4). A 7B model requires ~5-6 GB of VRAM.
