from abc import ABC, abstractmethod
from typing import Any, List, Union, Callable, Optional
import requests
import json


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
    def __init__(self, api_key: str, model: str = "sonar-pro"):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.perplexity.ai"
        self.endpoint = f"{self.base_url}/chat/completions"

    def ask(self, prompt: str, use_tools: bool = True, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Send a prompt to the model and return the response."""
        
        # Prepare messages
        messages = []
        
        # Get system prompt from tools if available
        if use_tools and self._tools and self._tools.system_prompts:
            system_prompt = self._tools.get_combined_system_prompt()
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # Make API request
            response = requests.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            # Extract content from response
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                
                # Add tool info for debugging if tools are active
                if use_tools and self._tools and self._tools.active_tools:
                    tool_names = ", ".join(self._tools.list_active_tools())
                    return f"[Using tools: {tool_names}] {content}"
                else:
                    return content
            else:
                return "No response content received from API"
                
        except requests.exceptions.RequestException as e:
            return f"API Request Error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def execute_chain(self, chain_steps: List[Union[dict, Callable]], initial_input: Any = None) -> Any:
        """Execute a chain of steps, returning the collective result."""
        from .chains import execute_chain
        return execute_chain(chain_steps, initial_input=initial_input)
