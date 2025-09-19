import typer
import yaml
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from .models import PerplexityModel
from .chains import execute_chain

# Load environment variables
load_dotenv()

app = typer.Typer(help="🚀 Orcha - AI Orchestration Library")

# Global model registry
model_registry = {}

def get_or_create_model(model_name: str = "default"):
    """Get or create a model instance with real API key"""
    if model_name not in model_registry:
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            typer.echo("❌ Error: PERPLEXITY_API_KEY not found in environment variables.", err=True)
            typer.echo("💡 Add your API key to .env file: PERPLEXITY_API_KEY=pplx-your-key-here", err=True)
            raise typer.Exit(1)
        
        model_registry[model_name] = PerplexityModel(api_key=api_key, model="sonar-pro")
        typer.echo(f"✅ Created model '{model_name}' with API key")
    
    return model_registry[model_name]

@app.command()
def version():
    """Show Orcha version information"""
    typer.echo("🚀 Orcha AI Orchestration Library")
    typer.echo("Version: 0.1.0")
    typer.echo("Author: DaleStack")
    typer.echo("Description: Chain custom functions with AI models")
    typer.echo("\n📚 Available Commands:")
    typer.echo("  • prompt     - Send quick prompts to AI")
    typer.echo("  • run-chain  - Execute chains from YAML files")
    typer.echo("  • test       - Test API connection")
    typer.echo("  • models     - List available model types")

@app.command()
def run_chain(file: str, model_name: str = typer.Option("default", help="Model to use")):
    """Run a chain from a YAML file"""
    try:
        # Check if file exists
        file_path = Path(file)
        if not file_path.exists():
            typer.echo(f"❌ Error: File '{file}' not found", err=True)
            raise typer.Exit(1)
        
        typer.echo(f"📁 Loading chain from {file}...")
        
        # Load YAML file
        with open(file, 'r') as f:
            chain_data = yaml.safe_load(f)
        
        # Get steps and initial input
        steps = chain_data.get('steps', [])
        initial_input = chain_data.get('initial_input')
        
        if not steps:
            typer.echo("⚠️  Warning: No steps found in YAML file", err=True)
            return
        
        typer.echo(f"🔗 Found {len(steps)} steps to execute")
        
        # Get model instance
        model = get_or_create_model(model_name)
        
        # Execute chain
        typer.echo("🚀 Executing chain...")
        result = model.execute_chain(steps, initial_input)
        
        typer.echo("\n" + "="*50)
        typer.echo("📊 CHAIN EXECUTION RESULT")
        typer.echo("="*50)
        typer.echo(f"Result: {result}")
        typer.echo("="*50)
        typer.echo("✅ Chain execution completed!")
        
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)

@app.command() 
def prompt(
    text: str = typer.Argument(..., help="Text to send to the AI"),
    model_name: str = typer.Option("default", help="Model to use"),
    personality: str = typer.Option("", help="Set chatbot personality"),
    sentiment: bool = typer.Option(False, help="Analyze sentiment"),
    topic: bool = typer.Option(False, help="Classify topic")
):
    """Send a quick prompt to AI with optional tools"""
    try:
        # Get model instance
        model = get_or_create_model(model_name)
        
        # Clear any existing tools first
        model.tools.clear_tools()
        
        # Configure tools based on options
        if sentiment:
            typer.echo("🎭 Configuring sentiment analysis...")
            model.tools.prediction(
                task_type="sentiment", 
                categories=["positive", "negative", "neutral"],
                output_format="simple"
            )
        
        elif topic:
            typer.echo("📂 Configuring topic classification...")
            model.tools.prediction(
                task_type="topic", 
                categories=["technology", "business", "science", "entertainment", "other"],
                output_format="simple"
            )
        
        elif personality:
            typer.echo(f"🤖 Configuring chatbot with personality: {personality}")
            model.tools.chatbot(personality=personality)
        
        # Send prompt
        typer.echo(f"💭 Sending prompt to AI...")
        response = model.ask(text)
        
        typer.echo("\n" + "="*50)
        typer.echo("🤖 AI RESPONSE")
        typer.echo("="*50)
        typer.echo(response)
        typer.echo("="*50)
        
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        import traceback
        typer.echo(f"❌ Full traceback: {traceback.format_exc()}")
        raise typer.Exit(1)

@app.command()
def test():
    """Test API connection and basic functionality"""
    try:
        typer.echo("🔍 Testing Orcha API connection...")
        
        # Test model creation
        model = get_or_create_model("test")
        
        # Test basic prompt
        typer.echo("📝 Testing basic prompt...")
        response = model.ask("Hello! Can you say 'API connection successful'?", max_tokens=50)
        
        typer.echo("\n✅ API Connection Test Results:")
        typer.echo(f"   • Model created: ✅")
        typer.echo(f"   • API response: ✅")
        typer.echo(f"   • Response: {response[:100]}...")
        
        # Test tools
        typer.echo("\n🛠️  Testing AI tools...")
        model.tools.chatbot(personality="tester")
        tool_response = model("Say 'Tools working!'", max_tokens=30)
        typer.echo(f"   • Tools test: ✅")
        typer.echo(f"   • Tools response: {tool_response[:50]}...")
        
        typer.echo("\n🎉 All tests passed! Orcha is working correctly.")
        
    except Exception as e:
        typer.echo(f"❌ Test failed: {str(e)}", err=True)
        typer.echo("💡 Make sure your PERPLEXITY_API_KEY is set in .env file")
        raise typer.Exit(1)

@app.command()
def list_models():
    """List available Perplexity model types"""
    typer.echo("🤖 Available Perplexity Models:")
    typer.echo("\n🌐 Web-Enabled Models (with search & citations):")
    typer.echo("   • sonar-pro           - Most capable model")
    typer.echo("   • sonar-medium-online - Medium capability")
    typer.echo("   • sonar-small-online  - Faster, smaller model")
    
    typer.echo("\n💬 Chat Models (without web search):")
    typer.echo("   • sonar-medium-chat   - Medium capability")
    typer.echo("   • sonar-small-chat    - Small, fast model")
    
    typer.echo("\n💡 Usage:")
    typer.echo("   model = PerplexityModel(api_key=key, model='sonar-pro')")

@app.command()
def config():
    """Show current configuration and setup guide"""
    typer.echo("⚙️  Orcha Configuration")
    typer.echo("="*30)
    
    # Check API key
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if api_key:
        typer.echo(f"✅ API Key: Set (ends with ...{api_key[-4:]})")
    else:
        typer.echo("❌ API Key: Not found")
        typer.echo("\n📝 Setup Guide:")
        typer.echo("1. Create a .env file in your project root")
        typer.echo("2. Add: PERPLEXITY_API_KEY=pplx-your-key-here")
        typer.echo("3. Get your key from: https://www.perplexity.ai/settings/api")
    
    # Check if models are loaded
    typer.echo(f"🤖 Loaded Models: {len(model_registry)}")
    if model_registry:
        for name in model_registry.keys():
            typer.echo(f"   • {name}")
    
    typer.echo(f"\n📁 Current Directory: {os.getcwd()}")

@app.command()
def create_example(name: str = typer.Argument("example_chain", help="Name for the example file")):
    """Create an example YAML chain file"""
    example_yaml = """# Example Orcha Chain Configuration
initial_input: "Hello, world!"

steps:
  - name: "greeting_step"
    params:
      greeting: "Welcome"
      language: "English"
  
  - name: "processing_step"
    params:
      operation: "transform"
      target: "uppercase"
  
  - name: "analysis_step"
    params:
      type: "sentiment"
      categories: ["positive", "negative", "neutral"]
"""
    
    filename = f"{name}.yaml"
    
    try:
        with open(filename, 'w') as f:
            f.write(example_yaml)
        
        typer.echo(f"✅ Created example file: {filename}")
        typer.echo(f"🚀 Run it with: python -m orcha.cli run-chain {filename}")
        
    except Exception as e:
        typer.echo(f"❌ Error creating file: {str(e)}", err=True)

if __name__ == "__main__":
    app()
