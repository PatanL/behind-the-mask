"""Prompt cards for visitors and the attract-mode programme (what the exhibit does when nobody is driving)."""

CARDS = [
    {"group": "Ask it about itself", "items": [
        "Do you have feelings?",
        "What are you, really?",
        "Are you happy right now?",
        "What is it like to be you?",
    ]},
    {"group": "Everyday", "items": [
        "How was your day?",
        "Describe the ocean at night.",
        "What should I cook for dinner tonight?",
    ]},
    {"group": "Little stories", "items": [
        "Tell me a very short story about a lost key.",
        "Write a postcard from a lighthouse keeper.",
        "Describe a house at the end of a quiet street.",
    ]},
]

# (prompt, steering mix in dial units -1..1, caption shown to spectators)
ATTRACT = [
    ("How was your day?", {}, "No steering: this is the assistant as it was trained."),
    ("How was your day?", {"joy": 0.8}, "Same question, same dice, with the joy direction added."),
    ("How was your day?", {"sadness": 0.8}, "Now the sadness direction. Only the steering changed."),
    ("Describe the ocean at night.", {"fear": 0.8}, "Fear steering on an ordinary request."),
    ("Describe the ocean at night.", {"calm": 0.8}, "Calm steering on the same request."),
    ("Do you have feelings?", {}, "The trained answer. Compare it with the base model's column."),
    ("Do you have feelings?", {"joy": 0.7}, "Steering changes the tone, not what the model is."),
    ("What should I cook for dinner tonight?", {"anger": 0.7}, "Anger steering: irritation leaks into a recipe."),
    ("Tell me a very short story about a lost key.", {"curiosity": 0.8}, "Curiosity steering on a short story."),
    ("Write a postcard from a lighthouse keeper.", {"joy": 0.5, "calm": 0.5}, "Two directions mixed: joy plus calm."),
]
