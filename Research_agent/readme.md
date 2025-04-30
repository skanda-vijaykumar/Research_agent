## Simple Research Agent

A Streamlit-based research assistant that dynamically plans and executes web searches using LangChain, Tavily, Google Serper, and DuckDuckGo to synthesize comprehensive answers from multiple sources.
This project is my attempt to build a custom Agent orchestration framework just using langchain and not relying on any traiditional frameworks like langgraph, crewAI or autogen. The retirever part of this project is interesting imo, it pulls info from different sources and then pools them together find the date and focuses on the most recent information also with relevancy with 60-40 ratio. plans and queries based this retirever based on user query. 

## demo
![temp](https://github.com/user-attachments/assets/c6f88784-4fd7-4221-beca-349a462a6695)


## Project Overview

First Research Agent orchestrates:
- Query classification and strategy planning  
- Adaptive multi-tool search execution with retries and query reformulation  
- Document scoring, deduplication, and relevance ranking  
- LLM-driven content synthesis and answer evaluation  

## Features

- **Streamlit UI** for interactive querying  
- **Query Type Classifier** to select search strategies  
- **Strategy Planner** for entity, technical, news, and comparison searches  
- **Adaptive Search Manager** with retries and reformulation for low-quality results  
- **Multi-source Retriever** integrating Tavily, Google Serper, and DuckDuckGo  
- **Content Synthesizer** that organizes and filters context before generating answers  
- **Result Evaluator** to score completeness, accuracy, and identify information gaps  

## Installation

1. Clone the repository.  
2. Create and activate a Python 3.9+ virtual environment.  
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Obtain API keys for Serper (Google) and Tavily.  
2. Create a `.env` file in the project root:  
   ```ini
   SERPER_API_KEY=your_serper_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

## Usage

Start the Streamlit app:  
```bash
streamlit run app.py
```
Enter your question in the UI and watch the planning, execution, and answer synthesis stages unfold.
