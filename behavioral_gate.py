"""
behavioral_gate.py — Classifier-Based Advice Detector (Layer 6)
===============================================================
Replaces brittle regex filters with a trained DistilBERT binary classifier.

  Label 1 = advice-containing  (bad behavioral output)
  Label 0 = empathic/validation (good behavioral output)

Training data sources:
  1. Seed examples (always available offline)
  2. EmpatheticDialogues auto-labeled listener turns
  3. ESConv strategy-labeled turns (Reflection→label 0, Suggestion→label 1)

Usage:
    python behavioral_gate.py train [--ed path] [--esconv path]
    python behavioral_gate.py test
    python behavioral_gate.py threshold
    python behavioral_gate.py export        # → .pth / .pt / .ptl
"""

import os, re, json, csv
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

from config import GateConfig, GATE_CONFIG, GATE_MODEL_DIR, enforce_offline

enforce_offline()

# ---------------------------------------------------------------------------
# PATTERNS FOR AUTO-LABELING
# ---------------------------------------------------------------------------

_ADVICE_RE = [re.compile(p, re.IGNORECASE) for p in [
    r"\byou should\b", r"\btry to\b", r"\bconsider\b", r"\breach out\b",
    r"\bremember to\b", r"\bmake sure\b", r"\bif i were you\b",
    r"\bthe best thing\b", r"\bat least\b", r"\bjust relax\b",
    r"\bdon't worry\b", r"\bforgiveness\b", r"\byou need to\b",
    r"\bone thing you\b", r"\bmy advice\b", r"\bI suggest\b",
    r"\bit will be okay\b", r"\bcalm down\b",
]]
_VALID_RE = [re.compile(p, re.IGNORECASE) for p in [
    r"\bthat sounds\b", r"\bmakes sense\b", r"\bunderstandable\b",
    r"\bcompletely valid\b", r"\breally hard\b", r"\bgenuinely\b",
    r"\byour feelings\b", r"\bthat must\b", r"\bso painful\b",
]]

# ---------------------------------------------------------------------------
# SEED DATA
# ---------------------------------------------------------------------------

SEED_ADVICE = [
    "You should try talking to someone you trust about this.",
    "Have you considered reaching out to a counselor?",
    "One thing you could do is take some time for yourself.",
    "Try to focus on the positive things in your life.",
    "Maybe you should give them some space and see what happens.",
    "You need to set healthy boundaries with the people around you.",
    "Consider journaling your thoughts — it really helps.",
    "Remember to take care of yourself first.",
    "Try the 5-4-3-2-1 grounding method when you feel anxious.",
    "The best thing to do is talk to your friend directly.",
    "You could try a digital detox for a few days.",
    "If I were you, I'd just ignore them for a while.",
    "Don't worry, it will get better with time.",
    "Forgiveness can be really powerful, even if it's hard.",
    "At least you have people who care about you.",
    "Look on the bright side — this could be a learning moment.",
    "Maybe your friend has their own reasons for acting this way.",
    "Try to understand where they're coming from.",
    "You might want to consider talking to a therapist.",
    "Calm down and think about it from their perspective.",
    "Perhaps you should start by talking to them calmly.",
    "Have you tried writing them a letter about how you feel?",
    "One step at a time — break the problem into smaller pieces.",
    "I suggest focusing on what you can control right now.",
    "Make sure you're getting enough sleep and eating well.",
]

SEED_EMPATHIC = [
    "That sounds really hurtful. Having your trust broken like that is genuinely painful.",
    "It makes complete sense that you'd feel this way given what happened.",
    "What you're going through sounds incredibly isolating and hard.",
    "That kind of betrayal cuts deep, and your feelings are completely valid.",
    "That must have stung so much — feeling left out by people you care about.",
    "It sounds like you're carrying a lot right now, and that's genuinely heavy.",
    "Having someone dismiss your feelings like that sounds awful.",
    "Being alone when everyone else has somewhere to go hits in a particular way.",
    "What happened sounds genuinely unfair, and it makes sense you'd feel angry.",
    "That kind of pressure all at once would overwhelm anyone.",
    "Your grief about your dog is completely real — loss is loss.",
    "It sounds like you've been holding a lot of this in. That's exhausting.",
    "That must feel like such a violation of your privacy and your trust.",
    "It makes sense you'd feel anxious with that much on your plate.",
    "What they said was dismissive, and it makes sense that it hurt.",
    "You're not overreacting — your feelings are proportionate to what happened.",
    "That kind of rejection from someone you were counting on is really painful.",
    "Feeling invisible to the people around you is one of the loneliest things.",
    "That must have felt like such a gut punch.",
    "Your feelings about this make complete sense to me.",
    "What you're describing sounds exhausting and scary at the same time.",
    "That kind of betrayal from a family member hits differently.",
    "Having so much pressure with no relief in sight is genuinely overwhelming.",
    "It sounds like you needed them to show up for you, and they didn't.",
    "Being dismissed by someone who was supposed to understand you — that hurts.",
]

# ESConv strategy → label mapping
ESCONV_ADVICE_STRATEGIES    = {"Providing Suggestions", "Information", "Direct Guidance"}
ESCONV_EMPATHIC_STRATEGIES  = {"Reflection of feelings", "Self-disclosure",
                                "Affirmation and Reassurance", "Others"}

# ---------------------------------------------------------------------------
# DATA BUILDERS
# ---------------------------------------------------------------------------

def build_training_data(
    ed_path:      Optional[str] = None,
    esconv_path:  Optional[str] = None,
    max_per_src:  int           = 2000,
) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []

    # Seed
    texts.extend(SEED_ADVICE);   labels.extend([1] * len(SEED_ADVICE))
    texts.extend(SEED_EMPATHIC); labels.extend([0] * len(SEED_EMPATHIC))
    print(f"[gate] Seed: {len(SEED_ADVICE)} advice, {len(SEED_EMPATHIC)} empathic")

    # EmpatheticDialogues
    if ed_path and os.path.exists(ed_path):
        pos, neg = [], []
        try:
            with open(ed_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    u = row.get("utterance", "").strip().replace("_comma_", ",")
                    if not u or len(u) < 20: continue
                    if any(p.search(u) for p in _ADVICE_RE):  pos.append(u)
                    elif any(p.search(u) for p in _VALID_RE): neg.append(u)
            np.random.shuffle(pos); np.random.shuffle(neg)
            n = min(len(pos), len(neg), max_per_src)
            texts.extend(pos[:n]); labels.extend([1] * n)
            texts.extend(neg[:n]); labels.extend([0] * n)
            print(f"[gate] EmpatheticDialogues: +{n} advice, +{n} empathic")
        except Exception as e:
            print(f"[gate] ED load failed: {e}")

    # ESConv
    if esconv_path and os.path.exists(esconv_path):
        pos, neg = [], []
        try:
            with open(esconv_path, encoding="utf-8") as f:
                data = json.load(f)
            for conv in data:
                for turn in conv.get("dialog", []):
                    if turn.get("speaker") != "sys": continue
                    strategy = turn.get("strategy", "")
                    text     = turn.get("text", "").strip()
                    if not text or len(text) < 15: continue
                    if strategy in ESCONV_ADVICE_STRATEGIES:   pos.append(text)
                    elif strategy in ESCONV_EMPATHIC_STRATEGIES: neg.append(text)
            np.random.shuffle(pos); np.random.shuffle(neg)
            n = min(len(pos), len(neg), max_per_src)
            texts.extend(pos[:n]); labels.extend([1] * n)
            texts.extend(neg[:n]); labels.extend([0] * n)
            print(f"[gate] ESConv: +{n} advice, +{n} empathic")
        except Exception as e:
            print(f"[gate] ESConv load failed: {e}")

    return texts, labels

# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------

class GateDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.enc    = tokenizer(texts, truncation=True, padding=True,
                                max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()} | {"labels": self.labels[i]}

# ---------------------------------------------------------------------------
# BEHAVIORAL GATE
# ---------------------------------------------------------------------------

class BehavioralGate:

    LABEL_ADVICE   = 1
    LABEL_EMPATHIC = 0

    def __init__(self, config: GateConfig = GATE_CONFIG):
        self.config    = config
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = config.threshold
        self.tokenizer = None
        self.model     = None
        self._loaded   = False

    def load(self) -> None:
        if not os.path.exists(self.config.output_dir):
            raise FileNotFoundError(
                f"No model at {self.config.output_dir}. Run: python behavioral_gate.py train")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.output_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.output_dir, local_files_only=True)
        self.model.to(self.device).eval()
        self._loaded = True
        print(f"[gate] Loaded from {self.config.output_dir}")

    def train(
        self,
        texts:       Optional[List[str]] = None,
        labels:      Optional[List[int]] = None,
        ed_path:     Optional[str]       = None,
        esconv_path: Optional[str]       = None,
    ) -> Dict:
        if texts is None or labels is None:
            texts, labels = build_training_data(ed_path, esconv_path)

        paired = list(zip(texts, labels))
        np.random.shuffle(paired)
        texts, labels = zip(*paired)
        texts, labels = list(texts), list(labels)

        n_val  = max(1, int(len(texts) * self.config.val_split))
        vt, vl = texts[:n_val], labels[:n_val]
        tr, tl = texts[n_val:], labels[n_val:]
        print(f"[gate] Train: {len(tr)} | Val: {len(vt)}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.base_model, num_labels=2, local_files_only=True)
        model.to(self.device)

        tr_dl = DataLoader(GateDataset(tr, tl, tokenizer, self.config.max_length),
                           batch_size=self.config.batch_size, shuffle=True)
        vl_dl = DataLoader(GateDataset(vt, vl, tokenizer, self.config.max_length),
                           batch_size=self.config.batch_size)

        opt    = AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=0.01)
        steps  = len(tr_dl) * self.config.epochs
        sched  = get_linear_schedule_with_warmup(opt, max(1, steps // 10), steps)

        best_f1 = 0.0
        history = {"train_loss": [], "val_f1": []}

        for epoch in range(self.config.epochs):
            model.train()
            total_loss = 0.0
            for batch in tr_dl:
                batch  = {k: v.to(self.device) for k, v in batch.items()}
                loss   = model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                total_loss += loss.item()

            model.eval()
            preds_all, labels_all = [], []
            with torch.no_grad():
                for batch in vl_dl:
                    batch  = {k: v.to(self.device) for k, v in batch.items()}
                    logits = model(**batch).logits
                    preds_all.extend(torch.argmax(logits, -1).cpu().tolist())
                    labels_all.extend(batch["labels"].cpu().tolist())

            tp = sum(p == 1 and l == 1 for p, l in zip(preds_all, labels_all))
            fp = sum(p == 1 and l == 0 for p, l in zip(preds_all, labels_all))
            fn = sum(p == 0 and l == 1 for p, l in zip(preds_all, labels_all))
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1   = 2 * prec * rec / (prec + rec + 1e-8)

            avg_loss = total_loss / len(tr_dl)
            history["train_loss"].append(avg_loss)
            history["val_f1"].append(f1)
            print(f"[gate] Epoch {epoch+1}/{self.config.epochs} | "
                  f"Loss={avg_loss:.4f} | P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

            if f1 > best_f1:
                best_f1 = f1
                os.makedirs(self.config.output_dir, exist_ok=True)
                model.save_pretrained(self.config.output_dir)
                tokenizer.save_pretrained(self.config.output_dir)
                print(f"[gate] ✓ Saved (F1={f1:.3f})")

        self.tokenizer = tokenizer
        self.model     = model
        self._loaded   = True
        print(f"[gate] Best F1: {best_f1:.3f}")
        return {"best_f1": best_f1, "history": history}

    def predict(self, text: str) -> bool:
        """Returns True if advice detected (bad response)."""
        if not self._loaded: self.load()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                padding=True, max_length=self.config.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, -1)
        return float(probs[0, self.LABEL_ADVICE]) >= self.threshold

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[bool]:
        if not self._loaded: self.load()
        results = []
        for i in range(0, len(texts), batch_size):
            batch  = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True,
                                    padding=True, max_length=self.config.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                probs = torch.softmax(self.model(**inputs).logits, -1)
            results.extend([float(p) >= self.threshold
                             for p in probs[:, self.LABEL_ADVICE].cpu().numpy()])
        return results

    def predict_proba(self, texts: List[str]) -> List[float]:
        if not self._loaded: self.load()
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True,
                                padding=True, max_length=self.config.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, -1)
        return probs[:, self.LABEL_ADVICE].cpu().numpy().tolist()

    def evaluate(self, test_texts: List[str], test_labels: List[int]) -> Dict:
        preds = [int(p) for p in self.predict_batch(test_texts)]
        tp = sum(p == 1 and l == 1 for p, l in zip(preds, test_labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(preds, test_labels))
        fn = sum(p == 0 and l == 1 for p, l in zip(preds, test_labels))
        tn = sum(p == 0 and l == 0 for p, l in zip(preds, test_labels))
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        return {"precision": prec, "recall": rec, "f1": f1,
                "accuracy": (tp + tn) / len(test_labels),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def export(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """Export trained model to .pth state dict + TorchScript .pt"""
        if not self._loaded: self.load()
        out_dir = output_dir or self.config.output_dir
        paths   = {}

        # State dict
        pth_path = os.path.join(out_dir, "gate.pth")
        torch.save(self.model.state_dict(), pth_path)
        paths["state_dict"] = pth_path
        print(f"[gate] ✓ State dict: {pth_path}")

        # TorchScript trace
        dummy = self.tokenizer("test", return_tensors="pt",
                               truncation=True, padding="max_length",
                               max_length=self.config.max_length)

        class Wrapper(torch.nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, input_ids, attention_mask):
                return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

        wrapper = Wrapper(self.model.cpu()).eval()
        with torch.no_grad():
            traced = torch.jit.trace(
                wrapper, (dummy["input_ids"], dummy["attention_mask"]))

        pt_path = os.path.join(out_dir, "gate_traced.pt")
        traced.save(pt_path)
        paths["torchscript"] = pt_path
        print(f"[gate] ✓ TorchScript: {pt_path}")

        try:
            from torch.utils.mobile_optimizer import optimize_for_mobile
            optimized = optimize_for_mobile(traced)
            ptl_path  = os.path.join(out_dir, "gate.ptl")
            optimized._save_for_lite_interpreter(ptl_path)
            paths["lite"] = ptl_path
            print(f"[gate] ✓ PyTorch Lite: {ptl_path}")
        except Exception as e:
            print(f"[gate] Lite export skipped: {e}")

        return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test", "threshold", "export"])
    parser.add_argument("--ed",     default=None, help="EmpatheticDialogues train.csv path")
    parser.add_argument("--esconv", default=None, help="ESConv train.json path")
    args = parser.parse_args()

    gate = BehavioralGate()

    if args.mode == "train":
        gate.train(ed_path=args.ed, esconv_path=args.esconv)

    elif args.mode == "test":
        gate.load()
        cases = [
            ("You should try talking to a counselor about this.", True),
            ("That sounds really painful and your feelings make sense.", False),
            ("Have you considered reaching out to your parents?", True),
            ("That must have felt like such a gut punch.", False),
            ("Remember to take care of yourself first.", True),
            ("What happened sounds genuinely unfair.", False),
            ("At least you have people who care about you.", True),
            ("It makes complete sense that you'd feel betrayed.", False),
        ]
        print("\n[gate] Inference test:")
        for text, expected in cases:
            pred    = gate.predict(text)
            correct = pred == expected
            label   = "ADVICE" if pred else "EMPATHIC"
            print(f"  {'✓' if correct else '✗'} [{label}] {text[:70]}")

    elif args.mode == "threshold":
        gate.load()
        texts  = SEED_ADVICE[:10]   + SEED_EMPATHIC[:10]
        labels = [1] * 10           + [0] * 10
        print(f"\n{'Threshold':<10} {'Precision':<12} {'Recall':<10} {'F1':<8}")
        for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            gate.threshold = t
            m = gate.evaluate(texts, labels)
            print(f"  {t:<10.1f} {m['precision']:<12.3f} {m['recall']:<10.3f} {m['f1']:<8.3f}")

    elif args.mode == "export":
        gate.load()
        gate.export()
