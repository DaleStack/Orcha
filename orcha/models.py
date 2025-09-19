from abc import ABC, abstractmethod
from typing import Any, List, Union, Callable, Optional
from .chains import execute_chain

class BaseModel(ABC):
    @abstractmethod
    def ask(self, prompt: str) -> str:
        """Send a prompt to the model and return the response."""
        pass

    @abstractmethod
    def execute_chain(self, chain_steps: List[Union[dict, Callable]]) -> Any:
        """Execute a chain of steps, returning the collective result."""
        pass

class PerplexityModel(BaseModel):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def ask(self, prompt: str) -> str:
        # Mocked response for now
        return f"Mock response from {self.model} for prompt: {prompt}"

    def execute_chain(self, chain_steps: List[Union[dict, Callable]], initial_input: Any = None) -> Any:
        """Execute a chain of steps, returning the collective result."""
        return execute_chain(chain_steps, initial_input=initial_input)