"""
Configuration and environment setup for the research agent.
"""
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Load environment variables
load_dotenv()

# Environment variables
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# Initialize global LLM
def get_llm(temperature=0.1):
    """
    Create and return a configured language model instance.
    
    Args:
        temperature (float): Controls randomness in the model's output.
            Lower values make output more deterministic.
    
    Returns:
        ChatOllama: Configured language model instance
    """
    return ChatOllama(
        model="llama3.1:latest",
        temperature=temperature,
        disable_streaming=True,
        num_ctx=8146,
        top_p=0.95,
        top_k=10,
        cache=False,
        mirostat=2,
        keep_alive=False,
        num_predict=900
    )
