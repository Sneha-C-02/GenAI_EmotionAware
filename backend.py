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

        top_emotion = emo["top_emotions"][0]["label"]

        memory_text = {
            "summary": f"Current emotional state: {top_emotion}",
            "arc_direction": "stable",
            "state": top_emotion
       }

        response = (
            f"I can understand that you're experiencing {top_emotion}. "
            f"That sounds emotionally difficult and important to you. "
            f"Thank you for sharing what you're going through."
        )

        gate_probability = 0.0

        return {
            "response": response,
            "emotion": emo,
            "vad": vad,
            "docs": docs,
            "memory": memory_text,
            "gate_probability": gate_probability,
        }


if __name__ == "__main__":
    bot = EmotionAwareChatbot()

    result = bot.generate_response("I feel lonely.")

    print(result)
