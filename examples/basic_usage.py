import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from orcha import PerplexityModel

# Create model instance
model = PerplexityModel(api_key="test", model="sonar")

print("=== Basic Usage ===")
print(model.ask("Hello"))
print()

print("=== Chatbot Tool ===")
model.tools.chatbot(
    personality="friendly programming tutor",
    expertise="Python programming",
    conversation_style="encouraging"
)
print(model("How do I create a function in Python?"))
print()

print("=== Prediction Tool - Sentiment Analysis ===")
model.tools.clear_tools()  # Clear previous tools
model.tools.prediction(
    task_type="sentiment",
    categories=["positive", "negative", "neutral"],
    output_format="json"
)
print(model("I love this new library! It's so easy to use."))
print()

print("=== Prediction Tool - Topic Classification ===")
model.tools.clear_tools()
model.tools.prediction(
    task_type="topic",
    categories=["technology", "sports", "politics", "entertainment"],
    output_format="simple"
)
print(model("The latest smartphone has an amazing camera and fast processor."))
print()

print("=== Callable Model Usage ===")
model.tools.clear_tools()
model.tools.chatbot(personality="helpful assistant")

# Different ways to call the model
print("Method 1:", model.ask("What is Python?"))
print("Method 2:", model("What is Python?"))  # Callable syntax
print()

print("=== List Active Tools ===")
print("Active tools:", model.tools.list_active_tools())
