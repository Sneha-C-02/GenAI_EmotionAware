import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from config import LLM_MODEL_PATH
from emotion_model import EmotionModel
from rag_pipeline_v2 import EmotionAwareRAGv2
from episodic_memory import EpisodicMemory
from behavioral_gate import BehavioralGate


class EmotionAwareChatbot:

    def __init__(self):
        print("Loading Emotion Model...")
        self.emotion = EmotionModel()

        print("Loading RAG...")
        self.rag = EmotionAwareRAGv2()

        print("Loading Memory...")
        self.memory = EpisodicMemory()

        print("Loading Gate...")
        self.gate = BehavioralGate()
        self.gate.load()

        print("Loading Mistral...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_PATH,
            local_files_only=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_PATH,
            device_map="auto",
            quantization_config=bnb,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        self.model.eval()

        print("Backend Ready!")

    def analyze(self, user_text):
        emo = self.emotion.predict(user_text)[0]

        vad = (
            emo["valence"],
            emo["arousal"],
            emo["dominance"]
        )

        docs, reasoning = self.rag.retrieve(
            query_text=user_text,
            user_state_vad=vad,
            mode="vad_augmented"
        )

        return {
            "emotion": emo,
            "vad": vad,
            "docs": docs,
            "reasoning": reasoning
        }

    def generate_response(self, user_text):
        result = self.analyze(user_text)

        emo = result["emotion"]
        vad = result["vad"]
        docs = result["docs"]

        # --------------------------------------------------
        # MEMORY
        # --------------------------------------------------
        state_label = "general_distress"
        if emo["valence"] > 0:
            state_label = "positive_engagement"

        self.memory.add_turn(
            text=user_text,
            vad=vad,
            state_label=state_label
        )

        memory_context = self.memory.get_context_injection()
        traj = self.memory.get_trajectory()

        memory_dict_for_ui = {
            "summary": self.memory.get_session_summary() or "No summary available.",
            "arc_direction": traj.arc_direction,
            "dominant_state": traj.dominant_state
        }

        # --------------------------------------------------
        # PROMPT
        # --------------------------------------------------
        SYSTEM_PROMPT = """You are an empathetic emotional-support assistant.

Rules:
- Acknowledge emotions.
- Validate feelings.
- Be warm and supportive.
- Respond in 2-4 sentences.
- Do NOT give direct advice.
- Do NOT tell users what they should do.
- Avoid:
  * You should
  * You need to
  * You must
  * Try to
  * Consider doing

Focus on understanding emotions."""

        context = "\n".join(f"- {doc}" for doc in docs[:3])

        system = SYSTEM_PROMPT
        if memory_context:
            system += "\n\nConversation Context:\n" + memory_context

        system += "\n\nRelevant Emotional Examples:\n" + context

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.model.device)

        # --------------------------------------------------
        # GENERATION
        # --------------------------------------------------
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        response = response.replace("<|assistant|>", "").strip()

        # --------------------------------------------------
        # BEHAVIORAL GATE
        # --------------------------------------------------
        gate_probability = float(self.gate.predict_proba([response])[0])

        return {
            "response": response,
            "emotion": emo,
            "vad": vad,
            "docs": docs,
            "memory": memory_dict_for_ui,
            "gate_probability": gate_probability,
        }


if __name__ == "__main__":
    bot = EmotionAwareChatbot()
    result = bot.generate_response("I feel lonely.")
    print(result["response"])
