"""
Adaptive search components for the research agent.
"""
import re
from typing import Dict, Any, List
from datetime import datetime
from langchain_core.messages import HumanMessage
from langchain.schema import Document

from ..config import get_llm
from ..models.data_models import SearchStrategy, SearchState, QueryExecution
from .tool_set import ToolSet

class AdaptiveSearchManager:
    """
    Manages adaptive search across multiple search tools with retry logic.
    """
    def __init__(self, tool_set: ToolSet, max_retries=3):
        self.tool_set = tool_set
        self.max_retries = max_retries
        self.llm = get_llm(temperature=0.2)
    
    def execute_search(self, search_step: Dict[str, Any], query_execution: QueryExecution) -> SearchState:
        """
        Execute a search step using the appropriate tool.
        
        Args:
            search_step (Dict[str, Any]): The search step to execute
            query_execution (QueryExecution): The current query execution context
            
        Returns:
            SearchState: The result of the search
        """
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
        """
        Execute a search with retries and query reformulation.
        
        Args:
            search_step (Dict[str, Any]): The search step to execute
            query_execution (QueryExecution): The current query execution context
            
        Returns:
            SearchState: The best search result after retries
        """
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
        """
        Assess the quality of search results.
        
        Args:
            query (str): The query used for the search
            documents (List[Document]): The search results
            original_query (str): The original user query
            
        Returns:
            float: A quality score between 0 and 1
        """
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
        """
        Reformulate a query to improve search results.
        
        Args:
            search_step (Dict[str, Any]): The search step that was executed
            query_execution (QueryExecution): The current query execution context
            search_state (SearchState): The current search state
            
        Returns:
            str: A reformulated query
        """
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
