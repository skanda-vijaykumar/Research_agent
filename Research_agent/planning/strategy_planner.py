"""
Search planning components for the research agent.
"""
import re
from datetime import datetime
from typing import List, Dict, Any
from ..config import get_llm
from ..models.data_models import SearchStrategy

class StrategyPlanner:
    """
    Creates and revises search plans based on query classification.
    """
    def __init__(self):
        self.llm = get_llm(temperature=0.1)
    
    def create_search_plan(self, query: str, classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create a search plan based on query classification.
        
        Args:
            query (str): The original user query
            classification (Dict[str, Any]): Classification details from the query analyzer
            
        Returns:
            List[Dict[str, Any]]: A list of search steps to execute
        """
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
        """Create a search plan focused on entity information."""
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
        """Create a search plan for technical queries."""
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
        """Create a search plan for news and recent information queries."""
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
        """Create a search plan for comparison queries."""
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
        """Create a general search plan for other types of queries."""
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
        """
        Revise a search plan based on execution results.
        
        Args:
            original_plan: The original search plan
            executed_steps: Steps that have been executed
            results_quality: Quality scores for executed steps
            
        Returns:
            List[Dict[str, Any]]: The revised search plan
        """
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
