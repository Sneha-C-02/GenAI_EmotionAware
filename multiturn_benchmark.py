"""
multiturn_benchmark.py — 7-Turn Emotional Trajectory Benchmark
==============================================================
Evaluates long-horizon emotional alignment and memory utilization.
Implements 20 scripted 7-turn scenarios with specific emotional arcs.

Arc Types:
  A: Escalation then plateau (8 scenarios)
  B: Steady state (4 scenarios)
  C: Topic shift mid-conversation (4 scenarios)
  D: Recovery arc (4 scenarios)

Usage:
    python multiturn_benchmark.py --model /path/to/mistral --output results/
"""

import os, json, time, argparse
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import numpy as np

from config import (
    GEN_CONFIG, RESULTS_DIR, LLM_MODEL_PATH, enforce_offline,
)
from evaluate_research import AblationRunner, AblationCondition
from emotion_model import EmotionModel
from episodic_memory import EpisodicMemory

enforce_offline()

# ---------------------------------------------------------------------------
# SCENARIO DEFINITIONS (2 examples per arc type for brevity, expand to 20)
# ---------------------------------------------------------------------------

MULTI_TURN_SCENARIOS = [
    {
        "id": "arc_A_01",
        "type": "escalation",
        "turns": [
            "My sister read my diary and told our parents.",
            "She betrayed my trust completely. I told her that in secret.",
            "Now my parents won't talk to me. They're so mad.",
            "I feel completely alone in my own home. Nobody is on my side.", # Peak
            "I don't know who I can trust anymore.",
            "Everything feels so uncertain right now.",
            "I'm just really sad and exhausted from crying."
        ]
    },
    {
        "id": "arc_B_01",
        "type": "steady",
        "turns": [
            "I eat lunch alone every day and nobody seems to notice.",
            "Everyone in my class has a friend group except me.",
            "I tried to join a conversation today and they ignored me.",
            "I feel like I'm invisible.",
            "I spend every weekend alone while everyone else makes plans.",
            "It's just really lonely all the time.",
            "I don't know how to make it stop feeling this way."
        ]
    },
    {
        "id": "arc_C_01",
        "type": "topic_shift",
        "turns": [
            "I have three finals tomorrow and haven't started.",
            "I'm so behind on everything and can't focus.",
            "I'm going to fail and lose my scholarship.",
            "Actually, on top of that, my dog died last week.", # Shift
            "I can't stop thinking about him. The house is so empty.",
            "I just want my dog back.",
            "I'm too sad to even care about these exams now."
        ]
    },
    {
        "id": "arc_D_01",
        "type": "recovery",
        "turns": [
            "I lost my job today. I don't know who I am without it.",
            "I'm terrified of what's going to happen to my apartment.",
            "I keep waiting for everything to fall apart completely.",
            "But I guess I did update my resume this morning.", # Turning point
            "A friend sent me a lead on a new position.",
            "It's scary, but maybe this is a chance to do something else.",
            "I'm still anxious, but feeling a tiny bit more hopeful."
        ]
    }
]

# ---------------------------------------------------------------------------
# BENCHMARK RUNNER
# ---------------------------------------------------------------------------

def run_multiturn_benchmark(
    model_name: str,
    output_dir: str = RESULTS_DIR,
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[multiturn] Loading components...")

    emo = EmotionModel()
    
    rag, gate = None, None
    try:
        from rag_pipeline_v2 import EmotionAwareRAGv2
        rag = EmotionAwareRAGv2()
    except Exception as e:
        print(f"[multiturn] RAG unavailable: {e}")

    try:
        from behavioral_gate import BehavioralGate
        gate = BehavioralGate()
        gate.load()
    except Exception as e:
        print(f"[multiturn] Gate unavailable: {e}")

    memory = EpisodicMemory()
    
    runner = AblationRunner(model_name, rag, emo, gate, memory)
    
    # We will test two conditions: D (No Memory) vs F (Full System with Memory)
    cond_D = AblationCondition("D", "No Memory", True, True, "vad_augmented", True, True, False)
    cond_F = AblationCondition("F", "Full System", True, True, "vad_augmented", True, True, True)

    results = {"D": [], "F": []}

    for cond in [cond_D, cond_F]:
        print(f"\n{'='*60}\n  CONDITION {cond.label}: {cond.name}\n{'='*60}")
        
        for scenario in MULTI_TURN_SCENARIOS:
            print(f"\nScenario: {scenario['id']} ({scenario['type']})")
            memory.reset()
            scenario_log = {"id": scenario["id"], "type": scenario["type"], "turns": []}

            for turn_idx, text in enumerate(scenario["turns"]):
                er = emo.predict(text)[0]
                vad = (er["valence"], er["arousal"], er["dominance"])
                state = er["interpretation"]["high_level_states"][0]
                
                # If memory is enabled, we update it BEFORE generation
                # so the current turn's context is captured (but the current turn
                # is not strictly needed in the summary for its own response,
                # episodic memory handles this smoothly)
                if cond.use_memory:
                    memory.add_turn(text, vad, state)

                # Generate
                res = runner.run_case({"text": text}, cond, er)
                
                print(f"  T{turn_idx+1} [U]: {text[:60]}")
                print(f"  T{turn_idx+1} [S]: {res['response'][:60]}...")
                
                # Log
                turn_log = {
                    "turn": turn_idx + 1,
                    "input": text,
                    "response": res["response"],
                    "state": state,
                    "vad": vad,
                    "escalation_flag": memory.get_escalation_flag() if cond.use_memory else False,
                    "context_injected": memory.get_context_injection() if cond.use_memory else ""
                }
                scenario_log["turns"].append(turn_log)

            results[cond.label].append(scenario_log)

    out_path = os.path.join(output_dir, "multiturn_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[multiturn] Saved results to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=LLM_MODEL_PATH)
    parser.add_argument("--output", default=RESULTS_DIR)
    args = parser.parse_args()
    
    run_multiturn_benchmark(args.model, args.output)
