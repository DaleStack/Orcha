from typing import List, Dict, Any, Optional

class AITools:
    """Collection of AI tools and capabilities for LLM models"""
    
    def __init__(self, model):
        self.model = model
        self.system_prompts = {}
        self.active_tools = []
    
    def chatbot(self, 
                personality: str = "helpful and friendly", 
                context: str = "",
                conversation_style: str = "casual",
                expertise: str = "general"):
        """Configure model as a conversational chatbot"""
        
        system_prompt = f"""You are a {personality} chatbot assistant.
        
Personality: {personality}
Conversation Style: {conversation_style}
Expertise Area: {expertise}

{f"Additional Context: {context}" if context else ""}

Instructions:
- Be conversational and engaging
- Provide helpful and accurate responses
- Ask follow-up questions when appropriate
- Maintain the specified personality throughout the conversation
- Stay focused on the expertise area if specified
"""
        
        self.system_prompts['chatbot'] = system_prompt
        if 'chatbot' not in self.active_tools:
            self.active_tools.append('chatbot')
        
        return self
    
    def prediction(self, 
                   task_type: str = "classification",
                   categories: List[str] = None,
                   output_format: str = "json",
                   confidence_scores: bool = True):
        """Configure for prediction tasks like classification, sentiment analysis, etc."""
        
        if categories is None:
            categories = ["positive", "negative", "neutral"]
        
        categories_str = ", ".join(categories)
        
        if task_type.lower() == "classification":
            system_prompt = f"""You are a text classification model.

Task: Classify the input text into one of these categories: {categories_str}

Output Format: {output_format.upper()}
Include Confidence: {confidence_scores}

Instructions:
- Analyze the input text carefully
- Choose the most appropriate category from: {categories_str}
- If output_format is 'json', format as: {{"category": "chosen_category", "confidence": 0.95}}
- If output_format is 'simple', just return the category name
- Be precise and consistent in your classifications
"""
        
        elif task_type.lower() == "sentiment":
            system_prompt = f"""You are a sentiment analysis model.

Task: Analyze the sentiment of the input text
Categories: {categories_str}
Output Format: {output_format.upper()}
Include Confidence: {confidence_scores}

Instructions:
- Determine the overall emotional tone of the text
- Choose from: {categories_str}
- Consider context, tone, and emotional indicators
- If output_format is 'json', format as: {{"sentiment": "category", "confidence": 0.95}}
- If output_format is 'simple', just return the sentiment
"""
        
        elif task_type.lower() == "topic":
            system_prompt = f"""You are a topic classification model.

Task: Identify the main topic/theme of the input text
Possible Topics: {categories_str}
Output Format: {output_format.upper()}
Include Confidence: {confidence_scores}

Instructions:
- Identify the primary subject or theme
- Choose the most relevant topic from: {categories_str}
- Focus on the main content, not minor details
- If output_format is 'json', format as: {{"topic": "category", "confidence": 0.95}}
- If output_format is 'simple', just return the topic name
"""
        
        else:
            system_prompt = f"""You are a prediction model for {task_type} tasks.

Categories/Options: {categories_str}
Output Format: {output_format.upper()}
Include Confidence: {confidence_scores}

Instructions:
- Analyze the input according to the {task_type} task
- Provide predictions based on the available categories
- Be accurate and confident in your predictions
"""
        
        self.system_prompts['prediction'] = system_prompt
        if 'prediction' not in self.active_tools:
            self.active_tools.append('prediction')
        
        return self
    
    def clear_tools(self):
        """Clear all active tools and system prompts"""
        self.system_prompts.clear()
        self.active_tools.clear()
        return self
    
    def get_combined_system_prompt(self) -> str:
        """Combine all active system prompts"""
        if not self.system_prompts:
            return ""
        
        combined = "\n\n".join(self.system_prompts.values())
        return combined
    
    def list_active_tools(self) -> List[str]:
        """List currently active tools"""
        return self.active_tools.copy()
