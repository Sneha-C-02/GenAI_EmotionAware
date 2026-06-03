import torch

from emotion_model import EmotionModel
from rag_pipeline_v2 import EmotionAwareRAGv2
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import LLM_MODEL_PATH

print("Loading Emotion Model...")
emotion = EmotionModel()

print("Loading RAG...")
rag = EmotionAwareRAGv2()

print("Loading Mistral...")

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL_PATH,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_PATH,
    device_map="auto",
    quantization_config=bnb,
    torch_dtype=torch.float16,
    local_files_only=True,
)

print("Chatbot Ready!")

SYSTEM_PROMPT = """
You are an empathetic emotional-support assistant.

Acknowledge emotions.
Validate feelings.
Do not give direct advice.
Respond in 2-4 sentences.
"""

while True:

    user = input("\nUser: ")

    if user.lower() in ["exit", "quit"]:
        break

    emo = emotion.predict(user)[0]

    vad = (
        emo["valence"],
        emo["arousal"],
        emo["dominance"]
    )

    docs, reasoning = rag.retrieve(
        user,
        vad,
        mode="vad_augmented"
    )

    context = "\n".join(
        f"- {d}" for d in docs[:3]
    )

    system = SYSTEM_PROMPT + f"\n\nRelevant Context:\n{context}"

    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    prompt = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    print("\nBot:", response)
