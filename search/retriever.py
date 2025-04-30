"""
Search retrievers for the research agent.
"""
import re
import math
import numpy as np
from datetime import datetime
from dateutil.parser import parse
from itertools import chain
from typing import List, Optional
from pydantic import Field
from langchain.schema import Document, BaseRetriever
from langchain_ollama.embeddings import OllamaEmbeddings

from tavily import TavilyClient
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

class EnhancedMultiSearchRetriever(BaseRetriever):
    """
    A retriever that combines results from multiple search engines.
    """
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
        """Get documents from Tavily search."""
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
        """Get documents from Google Serper search."""
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
        """Get documents from DuckDuckGo search."""
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
        """Enhance a query based on the search type."""
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
        """Extract a date from document content if available."""
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
        """Compute relevance scores for documents."""
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
        """Get relevant documents for a query across multiple search engines."""
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
        """Async version of get_relevant_documents."""
        return self._get_relevant_documents(query)


class DummySearchRetriever(BaseRetriever):
    """A fallback retriever that returns empty results."""
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Return an empty list of documents."""
        return []
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents."""
        return []
