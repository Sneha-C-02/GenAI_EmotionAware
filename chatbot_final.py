import torch

from emotion_model import EmotionModel
from rag_pipeline_v2 import EmotionAwareRAGv2
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from config import LLM_MODEL_PATH


class EmotionAwareChatbot:

    def __init__(self):

        print("Loading Emotion Model...")
        self.emotion = EmotionModel()

        print("Loading RAG...")
        self.rag = EmotionAwareRAGv2()

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

        self.SYSTEM_PROMPT = """
You are an empathetic emotional-support assistant.

Acknowledge emotions.
Validate feelings.
Do not give direct advice.
Respond in 2-4 sentences.
"""

        print("Chatbot Ready!")

    def generate_response(self, user_text):

        emo = self.emotion.predict(user_text)[0]

        vad = (
            emo["valence"],
            emo["arousal"],
            emo["dominance"]
        )

        docs, reasoning = self.rag.retrieve(
            user_text,
            vad,
            mode="vad_augmented"
        )

        context = "\n".join(
            f"- {d}" for d in docs[:3]
        )

        system = (
            self.SYSTEM_PROMPT
            + f"\n\nRelevant Context:\n{context}"
        )

        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]

        prompt = self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

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

        response = self.tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()

        return {
            "response": response,
            "emotion": emo,
            "vad": vad,
            "docs": docs,
            "reasoning": reasoning,
            "memory": {
                "summary": "Memory disabled",
                "arc_direction": "stable",
                "state": "neutral",
            },
            "gate_probability": 0.0,
        }


if __name__ == "__main__":

    bot = EmotionAwareChatbot()

    while True:

        user = input("\nUser: ")

        if user.lower() in ["exit", "quit"]:
            break

        result = bot.generate_response(user)

        print("\nBot:", result["response"])
