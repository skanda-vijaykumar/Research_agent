import re
import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from ..config import get_llm

class QueryTypeClassifier:
    def __init__(self):
        self.llm = get_llm(temperature=0.0)
    
    def classify(self, query: str) -> Dict[str, Any]:
        """
        Analyze and classify a user query.
        
        Args:
            query (str): The user query to classify
            
        Returns:
            Dict[str, Any]: Classification details including query type, entities, etc.
        """
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
