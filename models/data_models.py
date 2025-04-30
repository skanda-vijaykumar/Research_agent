"""
Core data models for the research agent.
"""
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import Field, BaseModel
from datetime import datetime
import uuid
from langchain.schema import Document

class SearchStrategy(str, Enum):
    """Enum defining different search strategies."""
    GENERAL = "general"               
    SPECIFIC_ENTITY = "specific_entity" 
    NEWS = "news"                     
    TECHNICAL = "technical"          
    FACTUAL = "factual"               
    DEEP_DIVE = "deep_dive"           
    COMPARISON = "comparison"         
    HISTORICAL = "historical"          
    LATEST = "latest"                  

class SearchState(BaseModel):
    """
    Represents the state and results of a search operation.
    """
    query: str
    strategy: SearchStrategy
    timestamp: datetime = Field(default_factory=datetime.now)
    documents: List[Document] = Field(default_factory=list)
    result_quality: float = 0.0
    search_provider: str = ""
    
    def is_successful(self) -> bool:
        """
        Determine if the search was successful based on quality and results.
        
        Returns:
            bool: True if the search was successful, False otherwise
        """
        return self.result_quality > 0.6 and len(self.documents) > 0

class QueryExecution(BaseModel):
    """
    Tracks the full execution context of a user query.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_query: str
    rephrased_queries: List[str] = Field(default_factory=list)
    searches: List[SearchState] = Field(default_factory=list)
    reasoning_steps: List[str] = Field(default_factory=list)
    final_answer: Optional[str] = None
    sources: List[Document] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    def add_search(self, search_state: SearchState):
        """Add a search state to the execution context."""
        self.searches.append(search_state)
    
    def add_reasoning(self, reasoning: str):
        """Add a reasoning step to the execution context."""
        self.reasoning_steps.append(reasoning)
    
    def get_successful_searches(self) -> List[SearchState]:
        """Return all successful searches."""
        return [s for s in self.searches if s.is_successful()]
    
    def get_latest_search(self) -> Optional[SearchState]:
        """Return the most recent search state."""
        return self.searches[-1] if self.searches else None
    
    def finish(self, final_answer: str, sources: List[Document]):
        """Complete the query execution with a final answer."""
        self.final_answer = final_answer
        self.sources = sources
        self.end_time = datetime.now()
