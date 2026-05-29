"""
episodic_memory.py — Multi-Turn Emotional Memory System (Layer 2)
=================================================================
Tracks emotional state across conversation turns and injects
natural-language context into Layer 4 (constrained prompt construction).

Design principles:
  - Context injected as NL PROSE only (no structured headers → prevents template leakage)
  - Rolling window of 7 turns (configurable)
  - Trajectory computed on every new turn
  - Session summary regenerated every 3 turns
  - Escalation flag triggers hard crisis stop in Layer 5

Research metrics enabled:
  - Valence Coherence (VC)
  - Escalation Detection Rate (EDR)
  - Turn-Level EAS (tEAS)
  - Response Consistency (RC)

Usage:
    memory = EpisodicMemory()
    memory.add_turn("I feel alone.", (-0.7, 0.3, -0.5), "social_withdrawal")
    context = memory.get_context_injection()   # → NL prose for prompt
    flag    = memory.get_escalation_flag()     # → True if crisis
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import deque

from config import MemoryConfig, MEMORY_CONFIG


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    turn_index:  int
    text:        str
    vad:         Tuple[float, float, float]
    state_label: str
    timestamp:   Optional[float] = None


@dataclass
class EmotionalTrajectory:
    valence_arc:         List[float]                  = field(default_factory=list)
    arousal_arc:         List[float]                  = field(default_factory=list)
    dominance_arc:       List[float]                  = field(default_factory=list)
    state_sequence:      List[str]                    = field(default_factory=list)
    arousal_slope:       float                        = 0.0
    valence_slope:       float                        = 0.0
    escalation_flag:     bool                         = False
    arc_direction:       str                          = "stable"
    dominant_state:      str                          = "unknown"
    state_transitions:   List[Tuple[int, str, str]]   = field(default_factory=list)
    peak_arousal_turn:   int                          = -1
    peak_intensity_turn: int                          = -1
    n_turns:             int                          = 0


# ---------------------------------------------------------------------------
# STATE → NL DESCRIPTION
# ---------------------------------------------------------------------------

_STATE_NL: Dict[str, str] = {
    "relationship_conflict":  "distress over a relationship conflict",
    "social_withdrawal":      "feelings of isolation or loneliness",
    "stress_overload":        "being overwhelmed by stress or pressure",
    "uncertainty":            "anxiety about an uncertain situation",
    "interpersonal_conflict": "conflict or tension with someone close to them",
    "grief_loss":             "grief or loss",
    "grief/loss":             "grief or loss",
    "positive_engagement":    "something positive or hopeful",
    "unknown":                "difficult emotions",
}

def _state_nl(state: str) -> str:
    return _STATE_NL.get(state.lower().replace(" ", "_"), "difficult emotions")


# ---------------------------------------------------------------------------
# CORE CLASS
# ---------------------------------------------------------------------------

class EpisodicMemory:

    def __init__(self, config: MemoryConfig = MEMORY_CONFIG):
        self.config = config
        self._buffer: deque[TurnRecord] = deque(maxlen=config.window)
        self._turn_count      = 0
        self._session_summary = ""
        self._unresolved: List[str] = []

    # -------------------------------------------------------------------------

    def add_turn(
        self,
        text:        str,
        vad:         Tuple[float, float, float],
        state_label: str,
        timestamp:   Optional[float] = None,
    ) -> None:
        record = TurnRecord(self._turn_count, text, vad, state_label, timestamp)
        self._buffer.append(record)
        self._turn_count += 1

        if state_label not in self._unresolved and state_label != "positive_engagement":
            self._unresolved.append(state_label)

        if self._turn_count % self.config.summary_interval == 0:
            self._session_summary = self._build_session_summary()

    # -------------------------------------------------------------------------

    def get_trajectory(self) -> EmotionalTrajectory:
        if not self._buffer:
            return EmotionalTrajectory()

        turns = list(self._buffer)
        val_arc   = [t.vad[0] for t in turns]
        aro_arc   = [t.vad[1] for t in turns]
        dom_arc   = [t.vad[2] for t in turns]
        state_seq = [t.state_label for t in turns]

        aro_slope = self._slope(aro_arc)
        val_slope = self._slope(val_arc)

        escalation = (
            len(turns) >= self.config.escalation_min_turns
            and aro_slope > self.config.escalation_arousal_slope
            and val_arc[-1] < self.config.escalation_valence_ceil
        )

        if val_slope > self.config.arc_improving_threshold:
            arc_dir = "improving"
        elif val_slope < self.config.arc_declining_threshold:
            arc_dir = "declining"
        else:
            arc_dir = "stable"

        dominant = max(set(state_seq), key=state_seq.count)

        transitions = []
        for i in range(1, len(turns)):
            if state_seq[i] != state_seq[i - 1]:
                transitions.append((i, state_seq[i - 1], state_seq[i]))

        intensities         = [abs(v) + a for v, a in zip(val_arc, aro_arc)]
        peak_aro_turn       = int(np.argmax(aro_arc))
        peak_intensity_turn = int(np.argmax(intensities))

        return EmotionalTrajectory(
            valence_arc         = val_arc,
            arousal_arc         = aro_arc,
            dominance_arc       = dom_arc,
            state_sequence      = state_seq,
            arousal_slope       = float(aro_slope),
            valence_slope       = float(val_slope),
            escalation_flag     = escalation,
            arc_direction       = arc_dir,
            dominant_state      = dominant,
            state_transitions   = transitions,
            peak_arousal_turn   = peak_aro_turn,
            peak_intensity_turn = peak_intensity_turn,
            n_turns             = len(turns),
        )

    # -------------------------------------------------------------------------

    def get_context_injection(self) -> str:
        """
        NL prose context for Layer 4 prompt. PROSE ONLY — no structured headers.
        Returns empty string if fewer than 2 turns in buffer.
        """
        if len(self._buffer) < 2:
            return ""

        traj   = self.get_trajectory()
        parts  = []
        dominant = _state_nl(traj.dominant_state)
        parts.append(f"The person has been expressing {dominant} across this conversation.")

        if traj.arc_direction == "improving":
            parts.append("Their emotional tone has been gradually improving.")
        elif traj.arc_direction == "declining":
            parts.append("Their distress appears to be intensifying.")

        if len(self._unresolved) > 1:
            concerns = " and ".join(_state_nl(s) for s in self._unresolved[-2:])
            parts.append(f"They haven't yet found resolution around {concerns}.")

        if traj.peak_intensity_turn < traj.n_turns - 1:
            peak_label = traj.state_sequence[traj.peak_intensity_turn]
            parts.append(
                f"The most intense moment was earlier when they described {_state_nl(peak_label)}."
            )

        return " ".join(parts)

    def get_session_summary(self) -> str:
        if not self._session_summary and self._buffer:
            self._session_summary = self._build_session_summary()
        return self._session_summary

    def get_escalation_flag(self) -> bool:
        if len(self._buffer) < self.config.escalation_min_turns:
            return False
        return self.get_trajectory().escalation_flag

    def get_retrieval_reranking_signal(self) -> Dict:
        """Signal for Layer 3 trajectory-aware reranking."""
        if not self._buffer:
            return {}
        traj    = self.get_trajectory()
        current = list(self._buffer)[-1]
        return {
            "arc_direction":  traj.arc_direction,
            "dominant_state": traj.dominant_state,
            "current_vad":    current.vad,
            "escalation":     traj.escalation_flag,
        }

    def reset(self) -> None:
        self._buffer.clear()
        self._turn_count      = 0
        self._session_summary = ""
        self._unresolved      = []

    def to_dict(self) -> Dict:
        traj = self.get_trajectory()
        return {
            "turn_count":        self._turn_count,
            "buffer_size":       len(self._buffer),
            "trajectory":        traj.__dict__,
            "session_summary":   self._session_summary,
            "unresolved":        self._unresolved,
            "context_injection": self.get_context_injection(),
        }

    # -------------------------------------------------------------------------

    def _slope(self, values: List[float]) -> float:
        n = len(values)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        y = np.array(values, dtype=float)
        xm, ym = x.mean(), y.mean()
        denom  = np.sum((x - xm) ** 2)
        return float(np.sum((x - xm) * (y - ym)) / denom) if denom > 1e-8 else 0.0

    def _build_session_summary(self) -> str:
        if not self._buffer:
            return ""
        turns = list(self._buffer)
        traj  = self.get_trajectory()
        s = (
            f"Session: {len(turns)} turn(s). "
            f"Primary concern: {_state_nl(traj.dominant_state)}. "
            f"Emotional arc: {traj.arc_direction}. "
        )
        if traj.escalation_flag:
            s += "⚠ ESCALATION DETECTED. "
        if self._unresolved:
            s += f"Unresolved: {', '.join(_state_nl(x) for x in self._unresolved[-3:])}."
        return s.strip()


# ---------------------------------------------------------------------------
# CLI DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    memory = EpisodicMemory()
    arc = [
        ("My sister read my diary and told our parents.",   (-0.5, 0.5, -0.3), "relationship_conflict"),
        ("She betrayed my trust completely.",               (-0.7, 0.6, -0.4), "relationship_conflict"),
        ("Now my parents won't talk to me.",                (-0.8, 0.7, -0.5), "social_withdrawal"),
        ("I feel completely alone in my own home.",         (-0.8, 0.6, -0.6), "social_withdrawal"),
        ("I don't know who I can trust anymore.",           (-0.6, 0.5, -0.5), "uncertainty"),
        ("Everything feels so uncertain.",                  (-0.6, 0.4, -0.4), "uncertainty"),
        ("I'm just really sad and exhausted.",              (-0.7, 0.3, -0.5), "grief_loss"),
    ]
    for i, (text, vad, state) in enumerate(arc):
        memory.add_turn(text, vad, state)
        print(f"\nTurn {i+1}: {text}")
        if i >= 1:
            print(f"  Context: {memory.get_context_injection()}")
        if memory.get_escalation_flag():
            print("  ⚠ ESCALATION FLAG ACTIVE")

    print(f"\nSummary: {memory.get_session_summary()}")
    traj = memory.get_trajectory()
    print(f"Arc direction: {traj.arc_direction}")
    print(f"Valence arc:   {[round(v, 2) for v in traj.valence_arc]}")
