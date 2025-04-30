"""
Search tools for the research agent.
"""
from ..config import get_llm, TAVILY_API_KEY, SERPER_API_KEY
from ..models.data_models import SearchStrategy
from .retriever import EnhancedMultiSearchRetriever, DummySearchRetriever

from tavily import TavilyClient
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

class ToolSet:
    """
    Manages the available search tools and retrievers.
    """
    def __init__(self):
        try:
            # Initialize all web search clients
            self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
            self.serper_client = GoogleSerperAPIWrapper(api_key=SERPER_API_KEY)
            self.ddg_search = DuckDuckGoSearchAPIWrapper()
            
            # Initialize search retrievers
            self.general_search = EnhancedMultiSearchRetriever(
                tavily_client=self.tavily_client,
                serper_client=self.serper_client,
                ddg_search=self.ddg_search,
                search_type="general"
            )
            
            self.entity_search = EnhancedMultiSearchRetriever(
                tavily_client=self.tavily_client,
                serper_client=self.serper_client,
                ddg_search=self.ddg_search,
                search_type="entity"
            )
            
            self.news_search = EnhancedMultiSearchRetriever(
                tavily_client=self.tavily_client,
                serper_client=self.serper_client,
                ddg_search=self.ddg_search,
                search_type="news"
            )
            
            self.technical_search = EnhancedMultiSearchRetriever(
                tavily_client=self.tavily_client,
                serper_client=self.serper_client,
                ddg_search=self.ddg_search,
                search_type="technical"
            )
        except Exception as e:
            print(f"Error initializing tools: {e}")
            # Create dummy tools if initialization fails
            self.general_search = DummySearchRetriever()
            self.entity_search = DummySearchRetriever()
            self.news_search = DummySearchRetriever()
            self.technical_search = DummySearchRetriever()
        
        # Reasoning tools
        self.llm = get_llm()
    
    def select_search_tool(self, strategy: SearchStrategy):
        """
        Select the appropriate search tool based on the search strategy.
        
        Args:
            strategy (SearchStrategy): The search strategy to use
            
        Returns:
            BaseRetriever: The selected search retriever
        """
        if strategy == SearchStrategy.SPECIFIC_ENTITY:
            return self.entity_search
        elif strategy in [SearchStrategy.NEWS, SearchStrategy.LATEST]:
            return self.news_search
        elif strategy in [SearchStrategy.TECHNICAL, SearchStrategy.DEEP_DIVE]:
            return self.technical_search
        else:
            return self.general_search  # Default
