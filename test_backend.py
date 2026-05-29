from backend import EmotionAwareChatbot

bot = EmotionAwareChatbot()

result = bot.analyze(
    "I feel lonely."
)

print(result)
