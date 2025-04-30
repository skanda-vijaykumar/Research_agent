import json
import re
from typing import Dict, Any, List
from datetime import date
from langchain.schema import Document
from langchain_core.messages import HumanMessage

from ..config import get_llm
from ..models.data_models import SearchState

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
