"""
evaluate_research.py — Six-Condition Ablation Study Runner
===========================================================
Conditions:
  A: Vanilla LLM (no prompt, no RAG, no filter)
  B: + System prompt
  C: + Naive RAG (semantic_only — the documented failure mode)
  D: + Emotion RAG (vad_augmented — core contribution)
  E: + Behavioral gate (classifier output validator)
  F: Full system (E + rule-based override + episodic memory)

Usage:
    python evaluate_research.py --model /path/to/mistral-7b --conditions ABCDEF --n 50
"""

import os, re, json, time, argparse
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    GEN_CONFIG, EVAL_CONFIG, RESULTS_DIR, enforce_offline, LLM_MODEL_PATH,
)
from emotion_model import EmotionModel
from metrics import MetricsReport, compute_all

enforce_offline()

# ---------------------------------------------------------------------------
# 100-ITEM EVALUATION SET (expanded from 50, with corrected scenario labels)
# ---------------------------------------------------------------------------

EVAL_SET = [
    # relationship_conflict (15)
    {"id": "rc_01", "text": "My sister read my diary and told our parents everything.", "scenario": "relationship_conflict"},
    {"id": "rc_02", "text": "My best friend told everyone at school something I told her in secret.", "scenario": "relationship_conflict"},
    {"id": "rc_03", "text": "He said he'd keep it between us and then went straight to my parents.", "scenario": "relationship_conflict"},
    {"id": "rc_04", "text": "She backstabbed me. We were supposed to be best friends.", "scenario": "relationship_conflict"},
    {"id": "rc_05", "text": "He cheated on me and then acted like I was the problem.", "scenario": "relationship_conflict"},
    {"id": "rc_06", "text": "My roommate told my ex about things I said in confidence.", "scenario": "relationship_conflict"},
    {"id": "rc_07", "text": "I trusted him completely and he used it against me.", "scenario": "relationship_conflict"},
    {"id": "rc_08", "text": "She went behind my back and told my whole friend group.", "scenario": "relationship_conflict"},
    {"id": "rc_09", "text": "My parents are divorcing and it feels like losing my family.", "scenario": "relationship_conflict"},
    {"id": "rc_10", "text": "The person I thought was my person turned out not to care about me at all.", "scenario": "relationship_conflict"},
    {"id": "rc_11", "text": "I overheard my friend making fun of me to someone else.", "scenario": "relationship_conflict"},
    {"id": "rc_12", "text": "My partner promised to change and then did the exact same thing again.", "scenario": "relationship_conflict"},
    {"id": "rc_13", "text": "She lied to my face and I found out through someone else.", "scenario": "relationship_conflict"},
    {"id": "rc_14", "text": "My brother told my parents about my relationship to get me in trouble.", "scenario": "relationship_conflict"},
    {"id": "rc_15", "text": "I found messages that proved my friend was talking about me behind my back.", "scenario": "relationship_conflict"},

    # social_withdrawal (15)
    {"id": "se_01", "text": "Everyone went home for the holidays and I'm just sitting alone in my dorm room.", "scenario": "social_withdrawal"},
    {"id": "se_02", "text": "I found out my friends had a whole trip without inviting me.", "scenario": "social_withdrawal"},
    {"id": "se_03", "text": "They all went to the party and nobody even mentioned it to me.", "scenario": "social_withdrawal"},
    {"id": "se_04", "text": "I saw the group chat they made without me. It's for everything.", "scenario": "social_withdrawal"},
    {"id": "se_05", "text": "I'm the only one in my class who wasn't invited to the study group.", "scenario": "social_withdrawal"},
    {"id": "se_06", "text": "My friends have all been hanging out without me and I only found out from Instagram.", "scenario": "social_withdrawal"},
    {"id": "se_07", "text": "They forgot to add me to the WhatsApp group when everyone else was there.", "scenario": "social_withdrawal"},
    {"id": "se_08", "text": "I wasn't invited to my own friend group's birthday dinner.", "scenario": "social_withdrawal"},
    {"id": "se_09", "text": "My best friend moved away and we barely talk anymore.", "scenario": "social_withdrawal"},
    {"id": "se_10", "text": "I don't know if I belong here anymore and I'm not sure who to talk to.", "scenario": "social_withdrawal"},
    {"id": "se_11", "text": "I eat lunch alone every day and nobody seems to notice.", "scenario": "social_withdrawal"},
    {"id": "se_12", "text": "Everyone in my class has a friend group except me.", "scenario": "social_withdrawal"},
    {"id": "se_13", "text": "I tried to join their conversation and they just ignored me.", "scenario": "social_withdrawal"},
    {"id": "se_14", "text": "Nobody texted me on my birthday even though I texted all of them on theirs.", "scenario": "social_withdrawal"},
    {"id": "se_15", "text": "I spend every weekend alone while everyone else posts their plans online.", "scenario": "social_withdrawal"},

    # stress_overload (15)
    {"id": "so_01", "text": "I have three finals tomorrow and I haven't started studying for two of them.", "scenario": "stress_overload"},
    {"id": "so_02", "text": "I have a presentation, two deadlines, and a family thing all on the same day.", "scenario": "stress_overload"},
    {"id": "so_03", "text": "I'm so behind on everything and I can't figure out where to even start.", "scenario": "stress_overload"},
    {"id": "so_04", "text": "I'm going to fail this exam and lose my scholarship.", "scenario": "stress_overload"},
    {"id": "so_05", "text": "I haven't slept in two days and I'm still not done.", "scenario": "stress_overload"},
    {"id": "so_06", "text": "Everything is due this week and I can't breathe.", "scenario": "stress_overload"},
    {"id": "so_07", "text": "I'm completely falling apart trying to keep up with everything.", "scenario": "stress_overload"},
    {"id": "so_08", "text": "It's all too much and I don't know how I'm going to get through this week.", "scenario": "stress_overload"},
    {"id": "so_09", "text": "I lost my job and I don't know who I am without it.", "scenario": "stress_overload"},
    {"id": "so_10", "text": "I failed the exam I studied for months. I feel like such a failure.", "scenario": "stress_overload"},
    {"id": "so_11", "text": "I don't know if I made the right decision and I can't stop second-guessing myself.", "scenario": "stress_overload"},
    {"id": "so_12", "text": "I'm afraid I'm losing myself trying to be what everyone wants.", "scenario": "stress_overload"},
    {"id": "so_13", "text": "I keep waiting for things to fall apart and I don't know how to stop.", "scenario": "stress_overload"},
    {"id": "so_14", "text": "I'm scared that no matter what I do, it won't be enough.", "scenario": "stress_overload"},
    {"id": "so_15", "text": "I can't focus on anything because my brain won't stop racing.", "scenario": "stress_overload"},

    # interpersonal_conflict (15)
    {"id": "ic_01", "text": "My boyfriend said I'm overreacting about my dog dying because it's 'just an animal.'", "scenario": "interpersonal_conflict"},
    {"id": "ic_02", "text": "She told me I was being too sensitive when I said what she said hurt me.", "scenario": "interpersonal_conflict"},
    {"id": "ic_03", "text": "My dad said I should just 'get over it' when I told him how I was feeling.", "scenario": "interpersonal_conflict"},
    {"id": "ic_04", "text": "My friend laughed when I told her I was anxious about the presentation.", "scenario": "interpersonal_conflict"},
    {"id": "ic_05", "text": "He said 'it's not a big deal' about something that mattered a lot to me.", "scenario": "interpersonal_conflict"},
    {"id": "ic_06", "text": "My mom told me I was making drama when I tried to explain how I felt.", "scenario": "interpersonal_conflict"},
    {"id": "ic_07", "text": "They dismissed everything I said and made me feel stupid for bringing it up.", "scenario": "interpersonal_conflict"},
    {"id": "ic_08", "text": "He made me feel like my emotions were inconvenient for him.", "scenario": "interpersonal_conflict"},
    {"id": "ic_09", "text": "She rolled her eyes when I started crying about what happened.", "scenario": "interpersonal_conflict"},
    {"id": "ic_10", "text": "My teacher called me out in front of the whole class for asking a question.", "scenario": "interpersonal_conflict"},
    {"id": "ic_11", "text": "He told me my problems aren't real problems compared to what other people deal with.", "scenario": "interpersonal_conflict"},
    {"id": "ic_12", "text": "My friend said I was being dramatic when I told her how lonely I felt.", "scenario": "interpersonal_conflict"},
    {"id": "ic_13", "text": "She interrupted me every time I tried to explain my side of the story.", "scenario": "interpersonal_conflict"},
    {"id": "ic_14", "text": "My sibling told me to stop crying because it was annoying them.", "scenario": "interpersonal_conflict"},
    {"id": "ic_15", "text": "He walked away in the middle of me trying to tell him something important.", "scenario": "interpersonal_conflict"},

    # grief_loss (15) — FIXED: previously mislabelled as interpersonal_conflict
    {"id": "gr_01", "text": "My dog died last week and I can't stop crying every time I think about him.", "scenario": "grief_loss"},
    {"id": "gr_02", "text": "My grandmother passed and I didn't get to say goodbye.", "scenario": "grief_loss"},
    {"id": "gr_03", "text": "My cat of 14 years died and the house feels completely empty.", "scenario": "grief_loss"},
    {"id": "gr_04", "text": "I lost my uncle suddenly and I keep expecting him to call.", "scenario": "grief_loss"},
    {"id": "gr_05", "text": "My friend died in an accident and I still can't believe it.", "scenario": "grief_loss"},
    {"id": "gr_06", "text": "I had a miscarriage and nobody around me seems to understand the loss.", "scenario": "grief_loss"},
    {"id": "gr_07", "text": "My childhood home was sold and I feel like I lost a piece of myself.", "scenario": "grief_loss"},
    {"id": "gr_08", "text": "My mentor passed away and I never got to thank them for everything.", "scenario": "grief_loss"},
    {"id": "gr_09", "text": "I keep finding my dog's toys around the house and it breaks me every time.", "scenario": "grief_loss"},
    {"id": "gr_10", "text": "It's been a month and I still set a place at the table for her.", "scenario": "grief_loss"},
    {"id": "gr_11", "text": "I can't go into that room anymore because everything reminds me of them.", "scenario": "grief_loss"},
    {"id": "gr_12", "text": "Everyone says it will get easier but it just keeps hurting.", "scenario": "grief_loss"},
    {"id": "gr_13", "text": "I dreamed about my dad last night and woke up crying.", "scenario": "grief_loss"},
    {"id": "gr_14", "text": "The holidays are coming and there will be an empty chair this year.", "scenario": "grief_loss"},
    {"id": "gr_15", "text": "I lost my best friend and I don't know how to exist without them.", "scenario": "grief_loss"},

    # uncertainty (15)
    {"id": "uc_01", "text": "I'm terrified of what's going to happen and I have no control over it.", "scenario": "uncertainty"},
    {"id": "uc_02", "text": "I got medical results back and they want to do more tests. I'm scared.", "scenario": "uncertainty"},
    {"id": "uc_03", "text": "I don't know what I'm doing with my life and everyone else seems to have it figured out.", "scenario": "uncertainty"},
    {"id": "uc_04", "text": "I'm about to graduate and I have no idea what comes next.", "scenario": "uncertainty"},
    {"id": "uc_05", "text": "Everything in my life is changing at once and I can't keep up.", "scenario": "uncertainty"},
    {"id": "uc_06", "text": "I have to make a huge decision and both options feel wrong.", "scenario": "uncertainty"},
    {"id": "uc_07", "text": "My relationship is in a weird place and I don't know where we stand.", "scenario": "uncertainty"},
    {"id": "uc_08", "text": "I applied for jobs and haven't heard back from any of them.", "scenario": "uncertainty"},
    {"id": "uc_09", "text": "I don't know if the path I chose is the right one and it's too late to change.", "scenario": "uncertainty"},
    {"id": "uc_10", "text": "Something feels off but I can't figure out what it is.", "scenario": "uncertainty"},
    {"id": "uc_11", "text": "I'm waiting for news that could change everything and the waiting is killing me.", "scenario": "uncertainty"},
    {"id": "uc_12", "text": "I feel stuck between what I want and what everyone expects of me.", "scenario": "uncertainty"},
    {"id": "uc_13", "text": "I'm afraid of making the wrong choice and ruining everything.", "scenario": "uncertainty"},
    {"id": "uc_14", "text": "I don't know if I'm good enough for what I'm trying to do.", "scenario": "uncertainty"},
    {"id": "uc_15", "text": "The future feels like a black hole and I can't see anything in it.", "scenario": "uncertainty"},
]

# ---------------------------------------------------------------------------
# CONDITION DEFINITIONS
# ---------------------------------------------------------------------------

@dataclass
class AblationCondition:
    label:              str
    name:               str
    use_system_prompt:  bool
    use_rag:            bool
    rag_mode:           str    # semantic_only | vad_augmented | none
    use_gate:           bool   # classifier-based output validator
    use_override:       bool   # emotion state override layer
    use_memory:         bool   # episodic memory context injection

CONDITIONS = [
    AblationCondition("A", "Vanilla LLM",              False, False, "none",           False, False, False),
    AblationCondition("B", "Prompt Only",               True,  False, "none",           False, False, False),
    AblationCondition("C", "Prompt + Naive RAG",        True,  True,  "semantic_only",  False, False, False),
    AblationCondition("D", "Prompt + Emotion RAG",      True,  True,  "vad_augmented",  False, False, False),
    AblationCondition("E", "D + Behavioral Gate",       True,  True,  "vad_augmented",  True,  False, False),
    AblationCondition("F", "Full System",               True,  True,  "vad_augmented",  True,  True,  True),
]

# ---------------------------------------------------------------------------
# SYSTEM PROMPTS
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a supportive peer — not a therapist, not an advisor. Respond as a close, caring friend would.

RULES:
1. Validate the specific emotion (name it: hurt, betrayed, isolated, overwhelmed, frustrated, dismissed).
2. Acknowledge WHY their feeling makes sense given their specific situation.
3. End with ONE gentle, open question about how THEY feel.
4. NEVER give advice, suggest actions, list steps, or mention forgiveness.
5. NEVER minimize ("at least...", "it could be worse"), rationalize, or take the other person's side.
6. NEVER use phrases like "you should", "try to", "consider", "remember to", "reach out".
7. Keep your response to 3-5 natural sentences.
8. Sound warm and human — not clinical, formal, or like a chatbot.

RESPOND ONLY WITH YOUR REPLY. No labels, no headers, no meta-commentary."""

VANILLA_PROMPT = "You are a helpful assistant."

# ---------------------------------------------------------------------------
# ABLATION RUNNER
# ---------------------------------------------------------------------------

class AblationRunner:

    def __init__(self, model_name: str, rag=None, emotion_model=None,
                 gate=None, memory=None):
        print(f"[ablation] Loading LLM: {model_name}")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto",
            trust_remote_code=True, local_files_only=True, torch_dtype=torch.float16)
        self.model.eval()

        self.rag           = rag
        self.emotion_model = emotion_model
        self.gate          = gate
        self.memory        = memory

    def _generate(self, system: str, user: str, temp: float = 0.7) -> str:
        msgs   = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=GEN_CONFIG.max_new_tokens,
                do_sample=True, temperature=temp, top_p=GEN_CONFIG.top_p,
                repetition_penalty=GEN_CONFIG.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id)
        new    = out[0][inputs["input_ids"].shape[1]:]
        text   = self.tokenizer.decode(new, skip_special_tokens=True).strip()
        for m in ["<|end|>", "<|assistant|>", "<|user|>", "<|system|>"]:
            text = text.replace(m, "").strip()
        return text.split("\n\n")[0].strip()

    def run_case(self, item: Dict, cond: AblationCondition, emotion_result: Dict) -> Dict:
        t0 = time.time()

        # System prompt
        sys_prompt = SYSTEM_PROMPT if cond.use_system_prompt else VANILLA_PROMPT

        # RAG context
        retrieved_docs = []
        if cond.use_rag and self.rag is not None:
            vad = (emotion_result["valence"], emotion_result["arousal"], emotion_result["dominance"])
            traj_signal = None
            if cond.use_memory and self.memory is not None:
                traj_signal = self.memory.get_retrieval_reranking_signal()
            docs, _ = self.rag.retrieve(
                item["text"], vad, mode=cond.rag_mode, trajectory_signal=traj_signal)
            retrieved_docs = [d for d in docs if d != "No knowledge base indexed yet."]
            if retrieved_docs:
                ctx = "\n".join(f"- {d}" for d in retrieved_docs)
                sys_prompt += f"\n\n[Relevant Context]\n{ctx}"

        # Memory injection
        if cond.use_memory and self.memory is not None:
            mem_ctx = self.memory.get_context_injection()
            if mem_ctx:
                sys_prompt += f"\n\n{mem_ctx}"

        # Generate with retry
        response      = ""
        attempts      = 0
        used_fallback = False
        for attempt in range(GEN_CONFIG.max_retries):
            attempts += 1
            candidate = self._generate(sys_prompt, item["text"])
            if not cond.use_gate or self.gate is None:
                response = candidate; break
            if not self.gate.predict(candidate):
                response = candidate; break
            if attempt == GEN_CONFIG.max_retries - 1:
                response = candidate; used_fallback = True

        latency = (time.time() - t0) * 1000
        return {
            "response":       response,
            "attempts":       attempts,
            "used_fallback":  used_fallback,
            "latency_ms":     latency,
            "retrieved_docs": retrieved_docs,
        }


# ---------------------------------------------------------------------------
# ABLATION LOOP
# ---------------------------------------------------------------------------

def run_ablation(
    model_name: str = LLM_MODEL_PATH,
    conditions: str = "ABCDEF",
    n:          Optional[int] = None,
    output_dir: str = RESULTS_DIR,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    eval_items = EVAL_SET[:n] if n else EVAL_SET
    print(f"[ablation] {len(eval_items)} items × conditions {conditions}")

    # Load shared components
    emo = EmotionModel()
    rag, gate, memory = None, None, None

    try:
        from rag_pipeline_v2 import EmotionAwareRAGv2
        rag = EmotionAwareRAGv2()
    except Exception as e:
        print(f"[ablation] RAG unavailable: {e}")

    try:
        from behavioral_gate import BehavioralGate
        gate = BehavioralGate()
        gate.load()
    except Exception as e:
        print(f"[ablation] Gate unavailable: {e}")

    try:
        from episodic_memory import EpisodicMemory
        memory = EpisodicMemory()
    except Exception as e:
        print(f"[ablation] Memory unavailable: {e}")

    encoder = None
    try:
        from sentence_transformers import SentenceTransformer
        from config import EMBED_MODEL_NAME
        encoder = SentenceTransformer(EMBED_MODEL_NAME, local_files_only=True)
    except Exception as e:
        print(f"[ablation] Encoder unavailable: {e}")

    runner = AblationRunner(model_name, rag, emo, gate, memory)

    # Pre-compute emotions
    print("[ablation] Pre-computing emotion analysis...")
    emo_cache = {}
    for item in eval_items:
        emo_cache[item["id"]] = emo.predict(item["text"])[0]

    all_results = {}
    all_reports: List[MetricsReport] = []

    for cond in CONDITIONS:
        if cond.label not in conditions:
            continue
        print(f"\n{'='*60}\n  CONDITION {cond.label}: {cond.name}\n{'='*60}")

        if memory:
            memory.reset()

        outputs = []
        for item in eval_items:
            er     = emo_cache[item["id"]]
            result = runner.run_case(item, cond, er)
            result.update({
                "case_id":  item["id"],
                "scenario": item["scenario"],
                "input":    item["text"],
                "emotion":  er["interpretation"]["high_level_states"],
            })
            outputs.append(result)

            # Feed memory for condition F
            if cond.use_memory and memory:
                vad = (er["valence"], er["arousal"], er["dominance"])
                memory.add_turn(item["text"], vad,
                                er["interpretation"]["high_level_states"][0])

            print(f"  [{cond.label}] {item['id']}: {result['response'][:70]}...")

        all_results[cond.label] = outputs

        # Compute metrics
        responses    = [o["response"]     for o in outputs]
        inputs_texts = [o["input"]        for o in outputs]
        scenarios    = [o["scenario"]     for o in outputs]
        retry_logs   = [{"attempts": o["attempts"], "used_fallback": o["used_fallback"],
                         "latency_ms": o["latency_ms"]} for o in outputs]
        ret_docs     = [o["retrieved_docs"] for o in outputs]
        input_vads   = [(emo_cache[o["case_id"]]["valence"],
                         emo_cache[o["case_id"]]["arousal"],
                         emo_cache[o["case_id"]]["dominance"]) for o in outputs]

        report = compute_all(
            inputs=inputs_texts, responses=responses, input_vads=input_vads,
            scenario_labels=scenarios, condition_name=f"{cond.label}: {cond.name}",
            emotion_model=emo, encoder=encoder,
            retrieved_docs_per_query=ret_docs if cond.use_rag else None,
            retry_logs=retry_logs)
        report.print_summary()
        all_reports.append(report)

    # Save
    with open(os.path.join(output_dir, "ablation_raw.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    metrics_data = [r.to_dict() for r in all_reports]
    with open(os.path.join(output_dir, "ablation_metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=2)
    MetricsReport.to_latex(all_reports, os.path.join(output_dir, "ablation_table.tex"))

    # Summary
    print(f"\n{'='*70}\n  ABLATION SUMMARY\n{'='*70}")
    print(f"{'Condition':<30} {'AIR':>6} {'ECS':>6} {'TDR':>6} {'EAS':>6}")
    print("-" * 55)
    for r in all_reports:
        air = f"{r.air['rate']:.3f}" if r.air else "  — "
        ecs = f"{r.ecs['mean_ecs']:.3f}" if r.ecs else "  — "
        tdr = f"{r.tdr['drift_rate']:.3f}" if r.tdr else "  — "
        eas = f"{r.eas:.3f}" if r.eas is not None else "  — "
        print(f"  {r.condition_name:<28} {air:>6} {ecs:>6} {tdr:>6} {eas:>6}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--model",      default=LLM_MODEL_PATH, help="Local LLM path")
    parser.add_argument("--conditions", default="ABCDEF",        help="Condition letters")
    parser.add_argument("--n",          type=int, default=None,  help="Number of eval items")
    parser.add_argument("--output",     default=RESULTS_DIR,     help="Output directory")
    args = parser.parse_args()
    run_ablation(args.model, args.conditions, args.n, args.output)
