from typing import List, Union, Callable, Any

def execute_chain(chain_steps: List[Union[dict, Callable]], initial_input: Any = None) -> Any:
    """
    Executes a sequence of steps in order.
    Each step can be:
      - dict: with keys 'name' and params, which should be handled by registered handlers (not implemented yet)
      - callable: a user-defined function that takes input and returns output
    
    Args:
        chain_steps: List of steps as dicts or callables.
        initial_input: Data passed to the first step.
        
    Returns:
        The output of the last step.
    """
    data = initial_input
    for step in chain_steps:
        if callable(step):
            data = step(data)  # Pass data to user function
        elif isinstance(step, dict):
            # For now, just mock a dict step response:
            name = step.get("name", "unknown")
            params = step.get("params", {})
            data = f"Executed step '{name}' with params {params} and input {data}"
        else:
            raise TypeError(f"Unsupported chain step type: {type(step)}")
    return data
