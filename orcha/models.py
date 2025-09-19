from abc import ABC, abstractmethod
from typing import Any, List, Union, Callable, Optional


class BaseModel(ABC):
    def __init__(self):
        self._tools = None
    
    @property
    def tools(self):
        """Access to AI tools for this model"""
        if self._tools is None:
            from .tools import AITools
            self._tools = AITools(self)
        return self._tools
    
    @abstractmethod
    def ask(self, prompt: str) -> str:
        """Send a prompt to the model and return the response."""
        pass

    @abstractmethod
    def execute_chain(self, chain_steps: List[Union[dict, Callable]], initial_input: Any = None) -> Any:
        """Execute a chain of steps, returning the collective result."""
        pass
    
    def __call__(self, prompt: str = None, **kwargs) -> str:
        """Make the model callable directly"""
        if prompt is None:
            raise ValueError("Prompt is required")
        return self.ask(prompt, **kwargs)


class PerplexityModel(BaseModel):
    def __init__(self, api_key: str, model: str):
        super().__init__()
        self.api_key = api_key
        self.model = model

    def ask(self, prompt: str, use_tools: bool = True) -> str:
        """Send a prompt to the model and return the response."""
        
        # Get system prompt from tools if available
        system_prompt = ""
        if use_tools and self._tools and self._tools.system_prompts:
            system_prompt = self._tools.get_combined_system_prompt()
        
        # Combine system prompt with user prompt
        if system_prompt:
            full_prompt = f"SYSTEM: {system_prompt}\n\nUSER: {prompt}"
        else:
            full_prompt = prompt
        
        # Mock response that shows system prompt is working
        if system_prompt:
            tool_names = ", ".join(self._tools.list_active_tools()) if self._tools else "none"
            return f"[Using tools: {tool_names}] Mock response from {self.model} for prompt: {prompt}"
        else:
            return f"Mock response from {self.model} for prompt: {prompt}"

    def execute_chain(self, chain_steps: List[Union[dict, Callable]], initial_input: Any = None) -> Any:
        """Execute a chain of steps, returning the collective result."""
        from .chains import execute_chain
        return execute_chain(chain_steps, initial_input=initial_input)
