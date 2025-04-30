"""
Main Streamlit application for the research agent.
"""
import streamlit as st
from .search.tool_set import ToolSet
from .execution.task_engine import TaskExecutionEngine

def main():
    """
    Main entry point for the Streamlit application.
    """
    try:
        st.set_page_config(page_title="Research Agent", page_icon="🔍", layout="wide")
        
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
