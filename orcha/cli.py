import typer
import yaml
from pathlib import Path
from typing import Optional

# Import your Orcha classes
from .models import PerplexityModel
from .chains import execute_chain

app = typer.Typer()

# Simple model registry for now
models = {
    "default": PerplexityModel(api_key="demo", model="sonar")
}

@app.command()
def run_chain(file: str):
    """Run a chain from a YAML file"""
    try:
        # Load YAML file
        with open(file, 'r') as f:
            chain_data = yaml.safe_load(f)
        
        # Get steps and initial input
        steps = chain_data.get('steps', [])
        initial_input = chain_data.get('initial_input')
        
        # Execute chain
        model = models["default"]
        result = model.execute_chain(steps, initial_input)
        
        typer.echo(f"Result: {result}")
        
    except Exception as e:
        typer.echo(f"Error: {str(e)}")

@app.command() 
def prompt(text: str, model: str = "sonar"):
    """Send a quick prompt to a model"""
    model_instance = models["default"]
    response = model_instance.ask(text)
    typer.echo(f"Response: {response}")

if __name__ == "__main__":
    app()
