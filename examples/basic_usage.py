from orcha import PerplexityModel

def custom_step(input_data):
    return f"Custom processing: {input_data}"

model = PerplexityModel(api_key="test", model="sonar")

# Test basic ask
result1 = model.ask("Hello")
print(result1)

# Test chain execution
chain_steps = [
    {"name": "step1", "params": {"key": "value"}},
    custom_step
]
result2 = model.execute_chain(chain_steps, initial_input="Test Initial Data")
print(result2)
