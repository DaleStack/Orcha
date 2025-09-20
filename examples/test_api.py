from dotenv import load_dotenv
import os
load_dotenv()
import re
from orcha import PerplexityModel

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Create your model with the API key from .env
model = PerplexityModel(api_key=PERPLEXITY_API_KEY, model="sonar-pro")

sample_text = "i love AI Engineering, programming and artificial intelligence"

# USER-DEFINED FUNCTIONS (Simple & Clear)


def add_excitement(text):
    """Add excitement to the text"""
    print(f"🎉 Step 2: Adding excitement...")
    print(text)
    return f"{text}!!!"

def ai_improve_text(text):
    """Use AI to improve the text"""
    print(f"🤖 Step 3: AI improving the text...")
    
    # Configure AI as writing assistant
    model.tools.clear_tools()
    model.tools.chatbot(personality="helpful writing assistant")
    
    # Ask AI to improve the text
    improved = model(f"Make this text more professional and engaging: {text}")
    return improved

def make_uppercase(text):
    """Convert text to uppercase"""
    print(f"📝 Final Step: Converting to uppercase...")
    return text.upper()

def add_final_touch(text):
    """Add a nice prefix"""
    print(f"✨ Step 4: Adding final touch...")
    return f"🚀 RESULT: {text}"



# ==============================================================================
# RUN THE SIMPLE CHAIN
# ==============================================================================

def run_simple_chain():
    """Run a simple 4-step processing chain"""
    
    print("🔗 Simple Text Processing Chain\n")
    print(f"📄 Original text: '{sample_text}'\n")
    
    # Define simple processing chain
    simple_chain = [        
        add_excitement,    # Custom function  
        ai_improve_text,   # AI function
        make_uppercase, 
        add_final_touch    # Custom function
    ]
    
    # Execute the chain
    result = model.execute_chain(simple_chain, initial_input=sample_text)
    
    print(f"\n{result}")
    print(f"\n✅ Done! Processed through {len(simple_chain)} steps.")
    
    return result

# Run it
if __name__ == "__main__":
    final_result = run_simple_chain()