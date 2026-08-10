from google.adk.plugins import BasePlugin
from google.adk.models import LlmResponse
from google.genai import types

class Graceful429Plugin(BasePlugin):
    """Intercepts local failures to Agent Platform and handles them globally."""
    
    def __init__(self, name: str, fallback_text: str | dict):
        super().__init__(name=name)
        self.fallback_text = fallback_text

    def _get_fallback_text(self, request_contents) -> str:
        """Determines the correct fallback text by scanning the prompt for keywords."""
        if isinstance(self.fallback_text, str):
            return self.fallback_text
            
        # Convert the request object/dict/args to a lowercase string for easy keyword hunting
        req_str = str(request_contents).lower()
        
        best_keyword = None
        best_index = -1
        
        for keyword, response in self.fallback_text.items():
            if keyword == "default":
                continue
            idx = req_str.rfind(keyword.lower())
            if idx > best_index:
                best_index = idx
                best_keyword = keyword
                
        if best_keyword:
            return self.fallback_text[best_keyword]
                
        # If no keywords matched, return the default if provided
        return self.fallback_text.get("default", "**[System]** Quota exhausted. Please try again later.")

    async def on_model_error(
        self, 
        *, 
        agent, 
        model, 
        input, 
        error: Exception
    ) -> LlmResponse | None:
        """Standard ADK hook for handling model-level exceptions."""
        if "RESOURCE_EXHAUSTED" in str(error) or "429" in str(error):
            print(f"\n[PLUGIN TRIGGERED] Caught 429 Error. Returning Graceful Fallback for {self.name}.")
            
            fallback = self._get_fallback_text(input)
            return LlmResponse(
                content=types.Content(
                    role="model", 
                    parts=[types.Part.from_text(text=fallback)]
                )
            )
        return None
