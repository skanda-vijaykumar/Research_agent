import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain.schema import Document
from langchain_core.messages import HumanMessage

from ..config import get_llm
from ..models.data_models import QueryExecution, SearchState
from ..analysis.query_analyzer import QueryTypeClassifier
from ..planning.strategy_planner import StrategyPlanner
from ..search.tool_set import ToolSet
from ..search.adaptive_search import AdaptiveSearchManager
from ..synthesis.content_synthesis import ContentSynthesizer, ResultEvaluator

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
                "search_strategy": "deep_dive",
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
    
    def _create_followup_query(self, original_query: str, synthesized_result: Dict[str, Any], evaluation: Dict[str, Any]) -> str:
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
