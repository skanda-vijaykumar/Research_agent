import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from tavily import TavilyClient
from langchain.schema import Document, BaseRetriever
from typing import List, Dict, Any, Tuple, Optional, Set, Union
from pydantic import Field, BaseModel
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import torch
import numpy as np
from itertools import chain
import os
import json
from functools import partial
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from sentence_transformers import SentenceTransformer
import torch
import re
from datetime import datetime
from dateutil.parser import parse
import math
from datetime import date
from dotenv import load_dotenv
from langchain_ollama.embeddings import OllamaEmbeddings
import time
import threading
import uuid
from collections import deque
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional, Set, Union, Deque
from langchain_core.messages import HumanMessage

load_dotenv()

SERPER_API_KEY = os.environ["SERPER_API_KEY"] 
TAVILY_API_KEY = os.environ['TAVILY_API_KEY']

# Initialize global LLM
def get_llm(temperature=0.1):
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

## SEARCH STRATEGY TYPES

class SearchStrategy(str, Enum):
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
    
    query: str
    strategy: SearchStrategy
    timestamp: datetime = Field(default_factory=datetime.now)
    documents: List[Document] = Field(default_factory=list)
    result_quality: float = 0.0
    search_provider: str = ""
    
    def is_successful(self) -> bool:
        return self.result_quality > 0.6 and len(self.documents) > 0

class QueryExecution(BaseModel):
    
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
        self.searches.append(search_state)
    
    def add_reasoning(self, reasoning: str):
        self.searches.append(reasoning)
    
    def get_successful_searches(self) -> List[SearchState]:
        return [s for s in self.searches if s.is_successful()]
    
    def get_latest_search(self) -> Optional[SearchState]:
        return self.searches[-1] if self.searches else None
    
    def finish(self, final_answer: str, sources: List[Document]):
        self.final_answer = final_answer
        self.sources = sources
        self.end_time = datetime.now()

## QUERY ANALYSIS AND TASK PLANNING 

class QueryTypeClassifier:
    
    def __init__(self):
        self.llm = get_llm(temperature=0.0)
    
    def classify(self, query: str) -> Dict[str, Any]:

        
        
        classification_prompt = (
            "Analyze the following query to determine its type and the best search strategies.\n\n"
            f"Query: {query}\n\n"
            "Please classify this query based on the following aspects:\n"
            "1. Query type (choose one): factual, entity-focused, comparison, technical, opinion, recent-news, historical\n"
            "2. Time sensitivity (choose one): very-recent, recent, timeless, historical\n"
            "3. Information depth needed (choose one): basic, moderate, comprehensive\n"
            "4. Primary entities (list all entities like people, companies, products, etc.)\n"
            "5. Best search strategies (choose 1-3): general, specific_entity, news, technical, factual, deep_dive, comparison, historical, latest\n"
            "6. Suggested keywords for search (list 3-5 important terms to search for)\n\n"
            "Respond with JSON using the format below:\n"
            "```json\n"
            "{\n"
            "  \"query_type\": \"entity-focused\",\n"
            "  \"time_sensitivity\": \"recent\",\n"
            "  \"depth\": \"moderate\",\n"
            "  \"entities\": [\"Entity1\", \"Entity2\"],\n"
            "  \"search_strategies\": [\"specific_entity\", \"news\"],\n"
            "  \"search_keywords\": [\"keyword1\", \"keyword2\", \"keyword3\"]\n"
            "}\n"
            "```")
        
        try:
            result = self.llm.invoke([HumanMessage(content=classification_prompt)])
            response_content = result.content
            
            # Extract JSON
            json_pattern = r'```(?:json)?([\s\S]*?)```'
            match = re.search(json_pattern, response_content)
            if match:
                json_str = match.group(1).strip()
            else:
                json_str = re.search(r'(\{[\s\S]*\})', response_content).group(1)
            
            return json.loads(json_str)
        except Exception as e:
            print(f"Error in query classification: {e}")
            
            # Fallback classification
            words = query.split()
            
            # Simple heuristic for entities (capitalized words)
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
            
            # Keywords (non-stopwords)
            stopwords = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being", 
                        "in", "on", "at", "to", "for", "with", "by", "about", "from", "of", "some", "any"}
            keywords = [w for w in words if w.lower() not in stopwords and len(w) > 3]
            
            return {
                "query_type": "entity-focused" if entities else "factual",
                "time_sensitivity": "recent",
                "depth": "moderate",
                "entities": entities,
                "search_strategies": ["specific_entity", "general"],
                "search_keywords": keywords[:5]  # Up to 5 keywords
            }

class StrategyPlanner:
    
    def __init__(self):
        self.llm = get_llm(temperature=0.1)
    
    def create_search_plan(self, query: str, classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        
        strategies = classification.get("search_strategies", ["general"])
        entities = classification.get("entities", [])
        keywords = classification.get("search_keywords", [])
        
        # Create a default plan based on classification
        if "specific_entity" in strategies and entities:
            # Start with entity search
            return self._create_entity_search_plan(query, entities, keywords)
        elif "technical" in strategies:
            # Technical query plan
            return self._create_technical_search_plan(query, keywords)
        elif "news" in strategies or "latest" in strategies:
            # Recent information plan
            return self._create_news_search_plan(query, entities, keywords)
        elif "comparison" in strategies:
            # Comparison search plan
            return self._create_comparison_search_plan(query, entities, keywords)
        else:
            # General plan for other queries
            return self._create_general_search_plan(query, keywords)
    
    def _create_entity_search_plan(self, query: str, entities: List[str], keywords: List[str]) -> List[Dict[str, Any]]:
        main_entity = entities[0] if entities else ""
        
        plan = [
            {
                "id": "entity-basic",
                "description": f"Find basic information about {main_entity}",
                "search_strategy": SearchStrategy.SPECIFIC_ENTITY,
                "query": f"{main_entity} {' '.join(keywords[:3])}",
                "priority": 1,
                "dependencies": []
            },
            {
                "id": "entity-details",
                "description": f"Find detailed information about {main_entity}",
                "search_strategy": SearchStrategy.DEEP_DIVE,
                "query": f"{main_entity} background profile details {' '.join(keywords[:2])}",
                "priority": 2,
                "dependencies": ["entity-basic"]
            }
        ]
        
        # Add recent information if we're looking for a company or person
        entity_lower = main_entity.lower()
        if any(term in entity_lower for term in ["company", "corporation", "inc", "organization"]) or len(main_entity.split()) <= 3:
            plan.append({
                "id": "entity-recent",
                "description": f"Find recent information about {main_entity}",
                "search_strategy": SearchStrategy.NEWS,
                "query": f"{main_entity} recent news latest developments {datetime.now().year}",
                "priority": 2,
                "dependencies": ["entity-basic"]
            })
        
        # Add association search if we have multiple entities
        if len(entities) > 1:
            plan.append({
                "id": "entity-association",
                "description": f"Find information about {main_entity}'s relationship with {entities[1]}",
                "search_strategy": SearchStrategy.SPECIFIC_ENTITY,
                "query": f"{main_entity} connection relationship {entities[1]}",
                "priority": 3,
                "dependencies": ["entity-basic", "entity-details"]
            })
        
        return plan
    
    def _create_technical_search_plan(self, query: str, keywords: List[str]) -> List[Dict[str, Any]]:
        return [
            {
                "id": "technical-overview",
                "description": f"Find overview information about {' '.join(keywords[:3])}",
                "search_strategy": SearchStrategy.TECHNICAL,
                "query": f"{' '.join(keywords[:3])} overview explanation",
                "priority": 1,
                "dependencies": []
            },
            {
                "id": "technical-details",
                "description": f"Find detailed technical information about {' '.join(keywords[:3])}",
                "search_strategy": SearchStrategy.DEEP_DIVE,
                "query": f"{' '.join(keywords[:3])} detailed technical explanation specification",
                "priority": 2,
                "dependencies": ["technical-overview"]
            },
            {
                "id": "technical-application",
                "description": f"Find practical applications of {' '.join(keywords[:3])}",
                "search_strategy": SearchStrategy.TECHNICAL,
                "query": f"{' '.join(keywords[:3])} application use case example",
                "priority": 3,
                "dependencies": ["technical-details"]
            }
        ]
    
    def _create_news_search_plan(self, query: str, entities: List[str], keywords: List[str]) -> List[Dict[str, Any]]:
        main_subject = entities[0] if entities else ' '.join(keywords[:3])
        current_year = datetime.now().year
        
        return [
            {
                "id": "news-recent",
                "description": f"Find recent news about {main_subject}",
                "search_strategy": SearchStrategy.NEWS,
                "query": f"{main_subject} latest news {current_year}",
                "priority": 1,
                "dependencies": []
            },
            {
                "id": "news-developments",
                "description": f"Find recent developments about {main_subject}",
                "search_strategy": SearchStrategy.LATEST,
                "query": f"{main_subject} recent developments update {current_year}",
                "priority": 1,
                "dependencies": []
            },
            {
                "id": "news-analysis",
                "description": f"Find analysis of recent news about {main_subject}",
                "search_strategy": SearchStrategy.DEEP_DIVE,
                "query": f"{main_subject} news analysis implications",
                "priority": 2,
                "dependencies": ["news-recent", "news-developments"]
            }
        ]
    
    def _create_comparison_search_plan(self, query: str, entities: List[str], keywords: List[str]) -> List[Dict[str, Any]]:
        if len(entities) >= 2:
            entity1, entity2 = entities[0], entities[1]
        else:
            comparison_pairs = re.findall(r'(\w+)\s+(?:vs\.?|versus|or|compared to)\s+(\w+)', query)
            if comparison_pairs:
                entity1, entity2 = comparison_pairs[0]
            else:
                # Just use keywords if no clear comparison
                entity1, entity2 = keywords[0], keywords[1] if len(keywords) > 1 else 'alternatives'
        
        return [
            {
                "id": f"compare-{entity1}",
                "description": f"Find information about {entity1}",
                "search_strategy": SearchStrategy.SPECIFIC_ENTITY,
                "query": f"{entity1} details features specifications",
                "priority": 1,
                "dependencies": []
            },
            {
                "id": f"compare-{entity2}",
                "description": f"Find information about {entity2}",
                "search_strategy": SearchStrategy.SPECIFIC_ENTITY,
                "query": f"{entity2} details features specifications",
                "priority": 1,
                "dependencies": []
            },
            {
                "id": "compare-direct",
                "description": f"Find direct comparisons between {entity1} and {entity2}",
                "search_strategy": SearchStrategy.COMPARISON,
                "query": f"{entity1} vs {entity2} comparison differences similarities",
                "priority": 2,
                "dependencies": [f"compare-{entity1}", f"compare-{entity2}"]
            }
        ]
    
    def _create_general_search_plan(self, query: str, keywords: List[str]) -> List[Dict[str, Any]]:
        # Extract key terms for searching
        search_terms = ' '.join(keywords[:4]) if keywords else query
        
        return [
            {
                "id": "general-search",
                "description": f"Find general information about {search_terms}",
                "search_strategy": SearchStrategy.GENERAL,
                "query": query,
                "priority": 1,
                "dependencies": []
            },
            {
                "id": "detailed-search",
                "description": f"Find detailed information about {search_terms}",
                "search_strategy": SearchStrategy.DEEP_DIVE,
                "query": f"{search_terms} detailed information explanation",
                "priority": 2,
                "dependencies": ["general-search"]
            }
        ]
    
    def revise_plan(self, original_plan: List[Dict[str, Any]], 
                   executed_steps: List[Dict[str, Any]], 
                   results_quality: Dict[str, float]) -> List[Dict[str, Any]]:

        # Clone the original plan
        revised_plan = original_plan.copy()
        
        # Get IDs of steps with poor results (quality < 0.6)
        poor_results = [step["id"] for step in executed_steps 
                      if step["id"] in results_quality and results_quality[step["id"]] < 0.6]
        
        # Create alternative steps for the poor-quality ones
        for step_id in poor_results:
            # Find the original step
            original_step = next((s for s in original_plan if s["id"] == step_id), None)
            if not original_step:
                continue
            
            # Create an alternative step with a different query formulation
            alternative = original_step.copy()
            alternative["id"] = f"{step_id}-alt"
            
            # Modify the query based on the strategy
            strategy = original_step.get("search_strategy", SearchStrategy.GENERAL)
            if strategy == SearchStrategy.SPECIFIC_ENTITY:
                # Try adding more specific terms for entity searches
                query_parts = alternative["query"].split()
                if len(query_parts) > 1:
                    alternative["query"] = f"{query_parts[0]} profile biography background information details"
            elif strategy == SearchStrategy.NEWS:
                # Try specifying different time ranges for news
                alternative["query"] = f"{original_step['query']} latest 2022 2023 2024"
            else:
                # For other strategies, add more general enhancement terms
                alternative["query"] = f"{original_step['query']} comprehensive guide explanation"
            
            # Add to plan
            revised_plan.append(alternative)
        
        return revised_plan

## ADAPTIVE SEARCH EXECUTION

class AdaptiveSearchManager:
    
    def __init__(self, tool_set, max_retries=3):
        self.tool_set = tool_set
        self.max_retries = max_retries
        self.llm = get_llm(temperature=0.2)
    
    def execute_search(self, search_step: Dict[str, Any], query_execution: QueryExecution) -> SearchState:

        # Get the appropriate search tool
        strategy = search_step.get("search_strategy", SearchStrategy.GENERAL)
        tool = self.tool_set.select_search_tool(strategy)
        
        # Prepare search query (either from step or do query reformulation)
        search_query = search_step.get("query", query_execution.original_query)
        if not search_query:
            search_query = query_execution.original_query
        
        # Remember this query
        if search_query not in query_execution.rephrased_queries:
            query_execution.rephrased_queries.append(search_query)
        
        # Execute search
        try:
            documents = tool._get_relevant_documents(search_query)
            
            # Create search state
            search_state = SearchState(
                query=search_query,
                strategy=strategy,
                documents=documents,
                search_provider=tool.__class__.__name__
            )
            
            # Assess quality
            search_state.result_quality = self._assess_result_quality(
                search_query, documents, query_execution.original_query
            )
            
            return search_state
            
        except Exception as e:
            print(f"Error executing search: {e}")
            # Return empty search state on error
            return SearchState(
                query=search_query,
                strategy=strategy,
                documents=[],
                result_quality=0.0,
                search_provider=tool.__class__.__name__
            )
    
    def execute_with_retries(self, search_step: Dict[str, Any], query_execution: QueryExecution) -> SearchState:
        best_search_state = None
        
        for attempt in range(self.max_retries):
            # Execute the search
            search_state = self.execute_search(search_step, query_execution)
            
            # Save the best result we've seen
            if best_search_state is None or search_state.result_quality > best_search_state.result_quality:
                best_search_state = search_state
            
            # If good enough, return it
            if search_state.result_quality > 0.7:
                return search_state
            
            # If not first attempt and still poor results, try query reformulation
            if attempt > 0 and search_state.result_quality < 0.5:
                reformulated_query = self._reformulate_query(search_step, query_execution, search_state)
                
                if reformulated_query != search_state.query:
                    # Create a modified search step with the new query
                    modified_step = search_step.copy()
                    modified_step["query"] = reformulated_query
                    
                    # Try again with reformulated query
                    search_state = self.execute_search(modified_step, query_execution)
                    
                    # Update best if better
                    if search_state.result_quality > best_search_state.result_quality:
                        best_search_state = search_state
        
        # Return the best search state we found
        return best_search_state
    
    def _assess_result_quality(self, query: str, documents: List[Document], original_query: str) -> float:

        if not documents:
            return 0.0
        
        # Check for empty content
        if all(len(doc.page_content.strip()) < 50 for doc in documents):
            return 0.1
        
        # Look for query term coverage
        query_terms = set(re.findall(r'\b[A-Za-z]{3,}\b', query.lower()))
        original_terms = set(re.findall(r'\b[A-Za-z]{3,}\b', original_query.lower()))
        combined_terms = query_terms.union(original_terms)
        
        # Count how many documents contain each term
        term_coverage = {}
        for term in combined_terms:
            term_coverage[term] = sum(1 for doc in documents if term in doc.page_content.lower())
        
        # Overall coverage score
        coverage_score = sum(min(count, 3) for count in term_coverage.values()) / (len(combined_terms) * 3)
        
        # Document diversity score (penalize if all docs are too similar)
        doc_texts = [doc.page_content for doc in documents]
        
        # Simple diversity measure - ratio of unique sentences to total
        all_sentences = []
        for text in doc_texts:
            all_sentences.extend([s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 10])
        
        diversity_score = len(set(all_sentences)) / max(1, len(all_sentences))
        
        # Combined quality score
        quality_score = (0.7 * coverage_score) + (0.3 * diversity_score)
        
        return min(1.0, quality_score)
    
    def _reformulate_query(self, search_step: Dict[str, Any], 
                          query_execution: QueryExecution,
                          search_state: SearchState) -> str:

        
        # Get strategy and current query
        strategy = search_step.get("search_strategy", SearchStrategy.GENERAL)
        current_query = search_state.query
        
        # Prepare context from previous searches
        context = []
        for prev_search in query_execution.searches:
            if prev_search.query != current_query: 
                context.append(f"Previous search: {prev_search.query}")
                context.append(f"Results quality: {prev_search.result_quality:.2f}")
                if prev_search.documents:
                    context.append(f"Found {len(prev_search.documents)} documents")
        
        # If we have some search results, include snippets
        snippets = []
        for doc in search_state.documents:
            snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            snippets.append(snippet)
        
        reformulation_prompt = (
            "Your task is to reformulate a search query to get better results.\n\n"
            f"Original user question: {query_execution.original_query}\n\n"
            f"Current search query: {current_query}\n\n"
            f"Search strategy: {strategy}\n\n"
        )
        
        if context:
            reformulation_prompt += f"Context from previous searches:\n{chr(10).join(context)}\n\n"
        
        if snippets:
            reformulation_prompt += (
                f"Current search returned {len(snippets)} results. Here are snippets:\n"
                f"{chr(10).join(snippets[:3])}\n\n"
            )
        else:
            reformulation_prompt += "Current search returned no useful results.\n\n"
        
        reformulation_prompt += (
            "Instructions:\n"
            "1. Analyze why the current search might not be yielding good results\n"
            "2. Reformulate the query to be more specific and targeted\n"
            "3. Add any missing key terms or concepts\n"
            "4. Remove any terms that might be causing confusion\n"
            "5. Return ONLY the reformulated query as plain text, nothing else\n\n"
            "Reformulated query:"
        )
        
        try:
            result = self.llm.invoke([HumanMessage(content=reformulation_prompt)])
            new_query = result.content.strip()
            
            # Clean up any extra text like "Reformulated query:" that the LLM might add
            new_query = re.sub(r'^.*?query:?\s*', '', new_query, flags=re.IGNORECASE)
            
            # Verify it's not too similar to the current query
            if new_query == current_query:
                # Add modification to ensure it's different
                strategy_terms = {
                    SearchStrategy.SPECIFIC_ENTITY: "profile details background information",
                    SearchStrategy.NEWS: "latest news recent developments",
                    SearchStrategy.TECHNICAL: "technical details specifications how works",
                    SearchStrategy.FACTUAL: "facts information data statistics",
                    SearchStrategy.DEEP_DIVE: "comprehensive analysis detailed explanation",
                    SearchStrategy.COMPARISON: "versus comparison differences similarities",
                    SearchStrategy.HISTORICAL: "history timeline development evolution",
                    SearchStrategy.LATEST: "newest latest current recent updates"
                }
                
                # Add strategy-specific terms
                current_parts = new_query.split()
                new_query = f"{new_query} {strategy_terms.get(strategy, '')}"
            
            return new_query
        except Exception as e:
            print(f"Error in query reformulation: {e}")
            # Fallback: add some strategy-specific terms to the query
            if strategy == SearchStrategy.SPECIFIC_ENTITY:
                return f"{current_query} profile information details background"
            elif strategy == SearchStrategy.NEWS:
                return f"{current_query} latest news {datetime.now().year}"
            else:
                return f"{current_query} detailed information guide"

## SEARCH TOOLS AND RETRIEVAL

class ToolSet:
    
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
        if strategy == SearchStrategy.SPECIFIC_ENTITY:
            return self.entity_search
        elif strategy in [SearchStrategy.NEWS, SearchStrategy.LATEST]:
            return self.news_search
        elif strategy in [SearchStrategy.TECHNICAL, SearchStrategy.DEEP_DIVE]:
            return self.technical_search
        else:
            return self.general_search  # Default

class EnhancedMultiSearchRetriever(BaseRetriever):

    tavily_client: TavilyClient = Field(...)
    serper_client: GoogleSerperAPIWrapper = Field(...)
    ddg_search: DuckDuckGoSearchAPIWrapper = Field(...)
    search_type: str = Field(default="general")
    encoder: OllamaEmbeddings = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.encoder = OllamaEmbeddings(
                model="nomic-embed-text",
            )
        except Exception as e:
            print(f"Error loading primary model: {e}")
            try:
                self.encoder = OllamaEmbeddings(
                    model="mxbai-embed-large",
                )
            except Exception as e:
                print(f"Error loading fallback model: {e}")
                self.encoder = None
                print("Falling back to basic text similarity")
    
    def _get_tavily_documents(self, query: str) -> List[Document]:
        try:
            # Configure search parameters based on search type
            search_depth = "advanced"
            max_results = 5
            include_answer = True
            search_params = {}
            
            if self.search_type == "news":
                search_params["search_depth"] = "advanced"
                search_params["include_domains"] = ["news.google.com", "reuters.com", "apnews.com", "bbc.com", "cnn.com", "nytimes.com"]
                search_params["max_results"] = 7
            elif self.search_type == "technical":
                search_params["search_depth"] = "advanced"
                search_params["max_results"] = 4
            elif self.search_type == "entity":
                search_params["include_answer"] = True
                search_params["max_results"] = 6
            
            # Enhance query based on search type
            enhanced_query = self._enhance_query(query)
            print(f"Enhanced Tavily query ({self.search_type}): {enhanced_query}")
            
            # Execute search
            response = self.tavily_client.search(
                query=enhanced_query, 
                search_depth=search_params.get("search_depth", search_depth),
                include_answer=search_params.get("include_answer", include_answer),
                max_results=search_params.get("max_results", max_results),
                include_domains=search_params.get("include_domains", None),
                exclude_domains=search_params.get("exclude_domains", None)
            )
            
            # Process results
            results = response.get("results", [])
            answer = response.get("answer", "")
            documents = []
            
            # Add Tavily-generated answer as a document if available
            if answer and len(answer) > 50:
                documents.append(
                    Document(
                        page_content=answer,
                        metadata={
                            "source": "Tavily Answer",
                            "title": "Tavily Generated Answer",
                            "provider": "Tavily",
                            "orig_order": 0  
                        }
                    )
                )
            
            # Add search results
            for res in results:
                if len(res.get("content", "")) < 20:
                    continue
                documents.append(
                    Document(
                        page_content=res.get("content", ""),
                        metadata={
                            "source": res.get("url", ""),
                            "title": res.get("title", ""),
                            "provider": "Tavily",
                            "orig_order": len(documents)
                        },
                    )
                )
            
            print(f"Got {len(documents)} documents from Tavily")
            return documents
        except Exception as e:
            print(f"Error getting Tavily results: {e}")
            return []
    
    def _get_serper_documents(self, query: str) -> List[Document]:
        try:
            # Enhance query based on search type
            enhanced_query = self._enhance_query(query)
            print(f"Enhanced Serper query ({self.search_type}): {enhanced_query}")
            
            # Configure search parameters
            search_params = {}
            
            if self.search_type == "news":
                raw_results = self.serper_client.results(enhanced_query, search_type="news")
                results = raw_results.get("news", [])
            else:
                raw_results = self.serper_client.results(enhanced_query)
                results = raw_results.get("organic", [])
            
            documents = []
            
            # Check for knowledgeGraph which has direct entity information
            knowledge_graph = raw_results.get("knowledgeGraph", {})
            if knowledge_graph and self.search_type == "entity":
                # Extract knowledge graph information
                kg_content = f"Title: {knowledge_graph.get('title', '')}\n"
                kg_content += f"Type: {knowledge_graph.get('type', '')}\n"
                for attr, value in knowledge_graph.items():
                    if attr not in ['title', 'type'] and isinstance(value, (str, int, float)):
                        kg_content += f"{attr}: {value}\n"
                
                # Add knowledge graph as a document
                if len(kg_content) > 50:
                    kg_doc = Document(
                        page_content=kg_content,
                        metadata={
                            "source": "Google Knowledge Graph",
                            "title": knowledge_graph.get('title', 'Knowledge Graph Result'),
                            "provider": "Google Serper",
                            "orig_order": 0  # Give it top priority
                        }
                    )
                    documents.append(kg_doc)
            
            # Add people also ask if available for entity searches
            if self.search_type == "entity" and "peopleAlsoAsk" in raw_results:
                paa_content = "People also ask:\n"
                for item in raw_results["peopleAlsoAsk"][:3]:
                    question = item.get("question", "")
                    answer = item.get("answer", "")
                    if question and answer:
                        paa_content += f"Q: {question}\nA: {answer}\n\n"
                
                if len(paa_content) > 100:
                    paa_doc = Document(
                        page_content=paa_content,
                        metadata={
                            "source": "Google People Also Ask",
                            "title": "Related Questions",
                            "provider": "Google Serper",
                            "orig_order": len(documents)
                        }
                    )
                    documents.append(paa_doc)
            
            # Add regular search results
            for res in results[:6]:  # Get up to 6 results
                content = res.get("snippet", "") if "snippet" in res else res.get("description", "")
                if len(content) < 20:
                    continue
                
                # For news, add date if available
                if self.search_type == "news" and "date" in res:
                    content = f"Date: {res['date']}\n{content}"
                
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": res.get("link", ""),
                            "title": res.get("title", ""),
                            "provider": "Google Serper",
                            "orig_order": len(documents)
                        },
                    )
                )
            
            print(f"Got {len(documents)} documents from Serper")
            return documents
        except Exception as e:
            print(f"Error getting Serper results: {e}")
            return []
    
    def _get_ddg_documents(self, query: str) -> List[Document]:
        try:
            # Enhance query based on search type
            enhanced_query = self._enhance_query(query)
            print(f"Enhanced DuckDuckGo query ({self.search_type}): {enhanced_query}")
            
            # Configure time for news searches
            if self.search_type == "news":
                enhanced_query += " time:m" 
            
            results = self.ddg_search.run(enhanced_query)
            if not results or len(results) < 20:
                return []
            
            # Try to split into individual results if possible
            try:
                # Split by common result separators
                splits = re.split(r'\n\s*\n|\[\d+\]:|^\d+\.\s', results)
                docs = []
                
                for i, split in enumerate(splits):
                    if len(split.strip()) > 50:
                        docs.append(
                            Document(
                                page_content=split.strip(),
                                metadata={
                                    "source": f"DuckDuckGo Result {i+1}",
                                    "title": f"DuckDuckGo Result {i+1}",
                                    "provider": "DuckDuckGo",
                                    "orig_order": i
                                }
                            )
                        )
                if docs:
                    return docs[:5]  # Keep top 5
            except Exception as e:
                print(f"Error splitting DuckDuckGo results: {e}")
            
            # Fallback to single document
            documents = [
                Document(
                    page_content=results,
                    metadata={
                        "source": "DuckDuckGo Search",
                        "title": "DuckDuckGo Results",
                        "provider": "DuckDuckGo",
                        "orig_order": 0
                    },
                )
            ]
            
            print(f"Got {len(documents)} documents from DuckDuckGo")
            return documents
        except Exception as e:
            print(f"Error getting DuckDuckGo results: {e}")
            return []
    
    def _enhance_query(self, query: str) -> str:
        # Extract main entity and concepts from query
        main_terms = []
        
        # Try to identify person names (capitalized words)
        name_candidates = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b', query)
        if name_candidates:
            main_terms.extend(name_candidates)
        
        # Try to identify organization names
        org_candidates = re.findall(r'\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b', query)
        for org in org_candidates:
            if org not in main_terms and len(org) > 3:  # Avoid short acronyms
                main_terms.append(org)
        
        # Build enhanced query based on search type
        if self.search_type == "entity":
            if any(name in query.lower() for name in name_candidates) and "First" in query.lower():
                return f"{query} profile LinkedIn position role employee company"
            elif any(name in query.lower() for name in name_candidates):
                return f"{query} profile biography information"
            elif any(org in query.lower() for org in org_candidates):
                return f"{query} company organization information details"
            else:
                return query
        
        elif self.search_type == "news":
            current_year = datetime.now().year
            return f"{query} news recent {current_year}"
        
        elif self.search_type == "technical":
            return f"{query} technical details explanation how works"
        
        else:
            # General search
            return query
    
    def _extract_date(self, content: str) -> Optional[datetime]:
        date_patterns = [
            r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{4}/\d{2}/\d{2}',
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2},?\s+\d{4}'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, content.lower(), re.IGNORECASE)
            if matches:
                try:
                    return parse(matches[0])
                except (ValueError, TypeError):
                    continue
        return None
    
    def _compute_scores(self, query: str, documents: List[Document]) -> List[tuple]:
        if not documents:
            return []
            
        current_date = datetime.now()
        
        try:
            if self.encoder is not None:
                query_embedding = self.encoder.embed_query(query)
                doc_embeddings = self.encoder.embed_documents(
                    [doc.page_content for doc in documents]
                )
                
                query_embedding = np.array(query_embedding)
                doc_embeddings = np.array(doc_embeddings)
                
                similarities = np.dot(doc_embeddings, query_embedding) / (
                    np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
                )
            else:
                from difflib import SequenceMatcher
                similarities = [
                    SequenceMatcher(None, query.lower(), doc.page_content.lower()).ratio() 
                    for doc in documents
                ]
                similarities = np.array(similarities)
            
            scored_docs = []
            for doc, similarity in zip(documents, similarities):
                doc_date = self._extract_date(doc.page_content)
                
                if doc_date:
                    days_old = (current_date - doc_date).days
                    recency_score = math.exp(-max(0, days_old) / 365)
                else:
                    recency_score = 0.5
                
                # Boost score for documents that mention specific terms from the query
                query_terms = set(re.findall(r'\b[A-Za-z]{3,}\b', query.lower()))
                doc_text = doc.page_content.lower()
                
                # Count how many query terms appear in the document
                term_matches = sum(1 for term in query_terms if term in doc_text)
                term_coverage = term_matches / max(1, len(query_terms))
                
                # Compute combined score with weights adjusted based on search type
                if self.search_type == "news":
                    combined_score = (0.2 * float(similarity)) + (0.6 * recency_score) + (0.2 * term_coverage)
                elif self.search_type == "entity":
                    combined_score = (0.3 * float(similarity)) + (0.2 * recency_score) + (0.5 * term_coverage)
                else:
                    combined_score = (0.4 * float(similarity)) + (0.2 * recency_score) + (0.4 * term_coverage)
                
                # Additional boosts based on source
                if "Knowledge Graph" in doc.metadata.get("source", ""):
                    combined_score *= 1.5
                elif doc.metadata.get("provider", "") == "Tavily" and doc.metadata.get("title", "") == "Tavily Generated Answer":
                    combined_score *= 1.3
                
                scored_docs.append((doc, combined_score))
                
                doc.metadata["similarity_score"] = float(similarity)
                doc.metadata["recency_score"] = float(recency_score)
                doc.metadata["term_coverage"] = float(term_coverage)
                doc.metadata["combined_score"] = float(combined_score)
            
            return sorted(scored_docs, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            print(f"Error in computing scores: {e}")
            return list(zip(documents, [1.0] * len(documents)))
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        tavily_docs = self._get_tavily_documents(query)
        serper_docs = self._get_serper_documents(query)
        ddg_docs = self._get_ddg_documents(query)
        
        all_docs = list(chain(tavily_docs, serper_docs, ddg_docs))
        
        if not all_docs:
            print(f"WARNING: No documents retrieved for query: {query}")
            return []
        
        seen_urls = set()
        unique_docs = []
        for doc in all_docs:
            url = doc.metadata["source"]
            if url not in seen_urls:
                seen_urls.add(url)
                unique_docs.append(doc)
        
        reranked_docs = self._compute_scores(query, unique_docs)
        return [doc for doc, _ in reranked_docs[:10]]
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

class DummySearchRetriever(BaseRetriever):
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        return []
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return []

## DYNAMIC TASK EXECUTION ENGINE

class TaskExecutionEngine:

    def __init__(self, tool_set: ToolSet):
        self.tool_set = tool_set
        self.query_classifier = QueryTypeClassifier()
        self.strategy_planner = StrategyPlanner()
        self.search_manager = AdaptiveSearchManager(tool_set)
        self.synthesizer = ContentSynthesizer()
        self.evaluator = ResultEvaluator()
        self.llm = get_llm()
    
    def process_query(self, query: str, streamlit_containers=None) -> Dict[str, Any]:

        # Create an execution context for this query
        execution = QueryExecution(original_query=query)
        
        # Set up streamlit containers if provided
        analysis_container = streamlit_containers.get("analysis") if streamlit_containers else None
        planning_container = streamlit_containers.get("planning") if streamlit_containers else None
        execution_container = streamlit_containers.get("execution") if streamlit_containers else None
        answer_container = streamlit_containers.get("answer") if streamlit_containers else None
        
        # Step 1: Classify the query
        if analysis_container:
            analysis_container.write("### Query Analysis")
            analysis_container.info("Analyzing query type...")
        
        classification = self.query_classifier.classify(query)
        
        if analysis_container:
            analysis_container.info(f"Query type: {classification.get('query_type', 'unknown')}")
            analysis_container.info(f"Time sensitivity: {classification.get('time_sensitivity', 'unknown')}")
            analysis_container.info(f"Depth needed: {classification.get('depth', 'moderate')}")
            
            entities = classification.get('entities', [])
            if entities:
                analysis_container.info(f"Entities: {', '.join(entities)}")
            
            search_strategies = classification.get('search_strategies', [])
            if search_strategies:
                analysis_container.info(f"Search strategies: {', '.join(search_strategies)}")
        
        # Step 2: Create initial search plan
        if planning_container:
            planning_container.write("### Search Planning")
            planning_container.info("Creating search plan...")
        
        search_plan = self.strategy_planner.create_search_plan(query, classification)
        
        if planning_container:
            for step in search_plan:
                planning_container.info(f"Search step: {step['description']}")
        
        # Step 3: Execute the search plan with dynamic adaptation
        if execution_container:
            execution_container.write("### Search Execution")
        
        # Track which steps have been executed and their quality
        executed_steps = []
        results_quality = {}
        documents_by_step = {}
        
        # Keep track of steps with dependencies
        pending_steps = []
        remaining_steps = search_plan.copy()
        
        # Execute independent steps first
        while remaining_steps or pending_steps:
            # Find a step with no dependencies or whose dependencies are satisfied
            next_step = None
            step_index = -1
            
            # First check remaining_steps
            for i, step in enumerate(remaining_steps):
                dependencies = step.get("dependencies", [])
                if not dependencies or all(dep in executed_steps for dep in dependencies):
                    next_step = step
                    step_index = i
                    break
            
            # If no step found, check pending_steps
            if next_step is None and pending_steps:
                for i, step in enumerate(pending_steps):
                    dependencies = step.get("dependencies", [])
                    if all(dep in executed_steps for dep in dependencies):
                        next_step = step
                        step_index = i
                        # Remove from pending
                        pending_steps.pop(i)
                        break
            
            # If still no step found, but there are remaining steps with unsatisfied dependencies
            if next_step is None and remaining_steps:
                # Move a step from remaining to pending
                next_step = remaining_steps[0]
                step_index = 0
                pending_steps.append(next_step)
            
            # If no steps found at all, we're done
            if next_step is None:
                break
            
            # Remove from remaining if found there
            if step_index >= 0 and remaining_steps:
                remaining_steps.pop(step_index)
            
            # Execute the step
            if execution_container:
                execution_container.info(f"Executing: {next_step['description']}")
            
            search_state = self.search_manager.execute_with_retries(next_step, execution)
            
            # Add the search state to the execution context
            execution.add_search(search_state)
            
            # Track step execution
            step_id = next_step["id"]
            executed_steps.append(step_id)
            results_quality[step_id] = search_state.result_quality
            documents_by_step[step_id] = search_state.documents
            
            if execution_container:
                if search_state.result_quality > 0.6:
                    execution_container.success(f"Completed: {next_step['description']} - Found {len(search_state.documents)} relevant documents")
                else:
                    execution_container.warning(f"Low quality results for: {next_step['description']} - Quality score: {search_state.result_quality:.2f}")
            
            # Check if we need to replan based on result quality
            if search_state.result_quality < 0.4 and len(executed_steps) < len(search_plan):
                if execution_container:
                    execution_container.warning("Low quality results detected. Revising search plan...")
                
                # Revise the plan
                revised_plan = self.strategy_planner.revise_plan(search_plan, [next_step], results_quality)
                
                # Add new steps from revised plan
                new_steps = [step for step in revised_plan if step["id"] not in executed_steps and step not in remaining_steps and step not in pending_steps]
                
                if new_steps:
                    if execution_container:
                        execution_container.info(f"Added {len(new_steps)} new search steps to the plan")
                    
                    # Add new steps to remaining
                    remaining_steps.extend(new_steps)
        
        # Step 4: Synthesize results into a final answer
        if answer_container:
            answer_container.write("### Answer Synthesis")
            answer_container.info("Synthesizing search results into a final answer...")
        
        # Collect all documents from successful searches
        all_docs = []
        for step_id, docs in documents_by_step.items():
            if results_quality.get(step_id, 0) > 0.3:  # Include even mediocre results
                all_docs.extend(docs)
        
        # Remove duplicate documents
        unique_docs = []
        seen_content = set()
        for doc in all_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        # Synthesize the final answer
        synthesized_result = self.synthesizer.synthesize(
            query=query,
            documents=unique_docs,
            search_history=execution.searches
        )
        
        # Evaluate the answer quality
        evaluation = self.evaluator.evaluate_result(synthesized_result, query, unique_docs)
        
        # Attempt one more search if quality is low
        if evaluation.get("quality_score", 0) < 0.6 and not execution.rephrased_queries:
            if answer_container:
                answer_container.warning("Initial answer quality is low. Performing additional research...")
            
            # Create a follow-up query based on identified gaps
            fallback_query = self._create_followup_query(query, synthesized_result, evaluation)
            
            # Execute the follow-up search
            fallback_step = {
                "id": "fallback-search",
                "description": f"Additional research for gaps in information",
                "search_strategy": SearchStrategy.DEEP_DIVE,
                "query": fallback_query,
                "priority": 1,
                "dependencies": []
            }
            
            search_state = self.search_manager.execute_with_retries(fallback_step, execution)
            
            # If follow-up search yielded good results, incorporate them
            if search_state.documents and search_state.result_quality > 0.4:
                if answer_container:
                    answer_container.info(f"Found additional information. Refining answer...")
                
                # Add new documents to our collection
                for doc in search_state.documents:
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        unique_docs.append(doc)
                
                # Re-synthesize with new information
                synthesized_result = self.synthesizer.synthesize(
                    query=query,
                    documents=unique_docs,
                    search_history=execution.searches
                )
                
                # Re-evaluate
                evaluation = self.evaluator.evaluate_result(synthesized_result, query, unique_docs)
        
        # Update the execution context with the final answer
        execution.finish(
            final_answer=synthesized_result["answer"],
            sources=unique_docs
        )
        
        # Display the final answer
        if answer_container:
            answer_container.success(synthesized_result["answer"])
            
            # Quality rating
            quality_score = evaluation.get("quality_score", 0) * 10
            answer_container.info(f"Answer quality rating: {quality_score:.1f}/10")
            
            # Display sources
            if unique_docs:
                sources_text = "Sources:\n"
                for i, doc in enumerate(unique_docs[:5]):  # Show top 5 sources
                    source = doc.metadata.get("source", "Unknown")
                    title = doc.metadata.get("title", "Unknown")
                    sources_text += f"{i+1}. {title} ({source})\n"
                
                answer_container.info(sources_text)
        
        return {
            "query": query,
            "answer": synthesized_result["answer"],
            "reasoning": synthesized_result.get("reasoning", ""),
            "sources": unique_docs,
            "quality_score": evaluation.get("quality_score", 0),
            "execution_context": execution
        }
    
    def _create_followup_query(self, original_query: str, 
                              synthesized_result: Dict[str, Any],
                              evaluation: Dict[str, Any]) -> str:
        
        gaps = evaluation.get("gaps", ["Missing specific details", "Incomplete information"])
        answer = synthesized_result.get("answer", "")
        
        followup_prompt = (
            "Based on an initial search for the following query, we need to create a follow-up search query to find missing information.\n\n"
            f"Original query: {original_query}\n\n"
            f"Initial answer: {answer[:500]}...\n\n"
            f"Information gaps identified: {', '.join(gaps)}\n\n"
            "Create a specific search query that will address these gaps and find the missing information. "
            "The query should be detailed and targeted to get precisely what's missing.\n\n"
            "Follow-up search query:"
        )
        
        try:
            result = self.llm.invoke([HumanMessage(content=followup_prompt)])
            followup_query = result.content.strip()
            
            # Clean up and make sure it's not too similar to the original
            followup_query = re.sub(r'^.*query:?\s*', '', followup_query, flags=re.IGNORECASE)
            
            if followup_query.lower() == original_query.lower():
                # Make sure it's different
                followup_query = f"{original_query} detailed information specifics background context"
            
            return followup_query
        except Exception as e:
            print(f"Error creating follow-up query: {e}")
            return f"{original_query} additional information detailed background"

## CONTENT SYNTHESIS AND EVALUATION

class ContentSynthesizer:
    
    def __init__(self):
        self.llm = get_llm(temperature=0.2)
    
    def synthesize(self, query: str, documents: List[Document], 
                  search_history: List[SearchState]) -> Dict[str, Any]:

        
        if not documents:
            return {
                "answer": f"I apologize, but I couldn't find specific information about '{query}'. This may be because the topic is very specialized, recent, or not widely documented online. Would you like me to suggest alternative search terms or approaches to find this information?",
                "reasoning": "No relevant documents were found after multiple search attempts."
            }
        
        # Prepare context from documents, prioritizing the most relevant ones
        top_docs = sorted(documents, key=lambda d: d.metadata.get("combined_score", 0), reverse=True)
        
        # Limit context to avoid token limits
        context_parts = []
        total_length = 0
        max_context_length = 16000  # Approximate token limit
        
        for doc in top_docs:
            doc_content = doc.page_content
            if total_length + len(doc_content) > max_context_length:
                # If adding this document would exceed limits, truncate it
                remaining_space = max_context_length - total_length
                if remaining_space > 200:  # Only add if we can include a meaningful chunk
                    doc_content = doc_content[:remaining_space] + "..."
                    context_parts.append(f"Source ({doc.metadata.get('provider', 'Unknown')} - {doc.metadata.get('source', 'Unknown')}): {doc_content}")
                    total_length += len(doc_content)
                break
            else:
                context_parts.append(f"Source ({doc.metadata.get('provider', 'Unknown')} - {doc.metadata.get('source', 'Unknown')}): {doc_content}")
                total_length += len(doc_content)
        
        # Generate the reasoning step (internal)
        today = date.today()
        reasoning_prompt = (
            "You are analyzing search results to prepare a comprehensive answer.\n\n"
            f"Today's date: {today}\n"
            f"Question: {query}\n\n"
            f"Context from search results:\n{chr(10).join(context_parts)}\n\n"
            "Instructions:\n"
            "1. Analyze what information is relevant to the question\n"
            "2. Identify any gaps or inconsistencies in the information\n"
            "3. Note what sources agree on and where they differ\n"
            "4. Consider the reliability and recency of information\n"
            "5. Organize the key points in a logical structure\n\n"
            "This reasoning will not be shown to the user but will help you organize your thoughts.\n"
            "Reasoning:"
        )
        
        try:
            reasoning_result = self.llm.invoke([HumanMessage(content=reasoning_prompt)])
            reasoning = reasoning_result.content if hasattr(reasoning_result, 'content') else str(reasoning_result)
            
            # Now generate the final answer
            answer_prompt = (
                "You are a helpful assistant that provides accurate and comprehensive answers based on search results.\n\n"
                f"Today's date: {today}\n"
                f"Question: {query}\n\n"
                "Based on your analysis of the search results, provide a clear, direct answer to the question.\n"
                "Your answer should be comprehensive yet concise, well-organized, and focused on the specific question asked.\n"
                "Do not mention the search process or refer to 'search results' or 'sources' in your answer.\n"
                "If the information is incomplete, acknowledge this honestly rather than speculating.\n\n"
                "Your answer:"
            )
            
            answer_result = self.llm.invoke([HumanMessage(content=answer_prompt)])
            answer = answer_result.content if hasattr(answer_result, 'content') else str(answer_result)
            
            return {
                "answer": answer,
                "reasoning": reasoning
            }
            
        except Exception as e:
            print(f"Error in content synthesis: {e}")
            
            # Fallback answer
            return {
                "answer": "Based on my search, I found some information related to your query, but I was unable to synthesize it properly due to a technical issue. The information suggests that " + 
                          (documents[0].page_content[:300] + "..." if documents else "there may be limited publicly available information on this topic."),
                "reasoning": f"Error during synthesis: {str(e)}"
            }

class ResultEvaluator:    
    def __init__(self):
        self.llm = get_llm(temperature=0.1)
    
    def evaluate_result(self, synthesis_result: Dict[str, Any], query: str, 
                       documents: List[Document]) -> Dict[str, Any]:

        
        if not documents:
            return {
                "quality_score": 0.2,
                "completeness": 0.1,
                "relevance": 0.5,
                "accuracy": 0.3,
                "gaps": ["No source documents found", "Unable to verify information"],
                "issues": ["Missing context", "Insufficient source material"]
            }
        
        answer = synthesis_result.get("answer", "")
        
        evaluation_prompt = (
            "Evaluate the quality of this answer to a user's question. Consider these criteria:\n\n"
            "1. Completeness: Does it fully address all aspects of the question?\n"
            "2. Relevance: Is the answer directly relevant to what was asked?\n"
            "3. Accuracy: Based on the source documents, does the answer appear accurate?\n"
            "4. Organization: Is the answer well-structured and easy to follow?\n"
            "5. Information Gaps: What important information is missing?\n\n"
            f"Question: {query}\n\n"
            f"Answer: {answer}\n\n"
            "Evaluate each criterion on a scale of 0.0 to 1.0, where 1.0 is perfect.\n"
            "Also identify any specific information gaps or issues with the answer.\n\n"
            "Respond with a JSON object with the following structure:\n"
            "```json\n"
            "{\n"
            "  \"completeness\": 0.8,\n"
            "  \"relevance\": 0.9,\n"
            "  \"accuracy\": 0.75,\n"
            "  \"organization\": 0.85,\n"
            "  \"quality_score\": 0.82,\n"
            "  \"gaps\": [\"missing specific dates\", \"no mention of important aspect X\"],\n"
            "  \"issues\": [\"slightly disorganized middle section\", \"could use more specific examples\"]\n"
            "}\n"
            "```"
        )
        
        try:
            result = self.llm.invoke([HumanMessage(content=evaluation_prompt)])
            response_content = result.content
            
            # Extract JSON
            json_pattern = r'```(?:json)?([\s\S]*?)```'
            match = re.search(json_pattern, response_content)
            if match:
                json_str = match.group(1).strip()
            else:
                json_str = re.search(r'(\{[\s\S]*\})', response_content).group(1)
            
            return json.loads(json_str)
        except Exception as e:
            print(f"Error in result evaluation: {e}")
            
            # Fallback evaluation
            return {
                "quality_score": 0.7,
                "completeness": 0.7,
                "relevance": 0.8,
                "accuracy": 0.7,
                "organization": 0.7,
                "gaps": ["Could not perform full evaluation due to technical error"],
                "issues": ["Evaluation process encountered an error"]
            }

## STREAMLIT APP

def main():
    try:
        st.set_page_config(page_title="First Research Agent", page_icon="🔍", layout="wide")
        
        # Set up the sidebar
        st.sidebar.image("static/bot8.png")
      
        
        # Set up the main header
        st.header("🔍 First Research Agent")
        st.markdown(
            "Ask any question and I'll search the web, analyze multiple sources, and provide a comprehensive answer."
        )
        
        # Create a clean area for the search input
        search_col, _ = st.columns([3, 1])
        with search_col:
            question = st.text_input("What would you like to know?", key="search_input")
        
        # Create containers for each stage of the process
        analysis_container = st.container()
        planning_container = st.container()
        execution_container = st.container()
        answer_container = st.container()
        
        # Initialize the execution engine
        tool_set = ToolSet()
        engine = TaskExecutionEngine(tool_set)
        
        # Process the query when submitted
        if question:
            with st.spinner('Researching your question...'):
                try:
                    # Set up containers for the execution engine
                    containers = {
                        "analysis": analysis_container,
                        "planning": planning_container,
                        "execution": execution_container,
                        "answer": answer_container
                    }
                    
                    # Process the query
                    result = engine.process_query(question, containers)
                    
                except Exception as e:
                    st.error(f"An error occurred while processing your question: {str(e)}")
                    st.info("You can try asking a different question or rephrasing your query.")
    except Exception as e:
        st.error(f"An error occurred while initializing the application: {str(e)}")
        st.info("Please try refreshing the page or contact the administrator if the issue persists.")

if __name__ == "__main__":
    main()
