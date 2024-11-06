import streamlit as st
import os, sys 
from typing import Dict, List  

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.singleton_config import ConfigSingleton
from src.paths import get_db_path, get_faiss_path

class RAGQueryUI:
    def __init__(self):
        self.config = ConfigSingleton()
        self.faiss_path = get_faiss_path()
        self.db_path = get_db_path()

    def setup_page(self):
        """Initialize page configuration."""
        st.set_page_config(
            page_title="RAG Query System",
            layout="wide",
            initial_sidebar_state="collapsed"
        )

    def render_query_input(self) -> Dict:
        """Render query input section as per design spec."""
        st.title("RAG Query System")
        
        # Query input section with parameters
        query_col, param_col, button_col = st.columns([3, 1, 1])
        
        with query_col:
            query = st.text_input("Enter your query:", key="main_query")
        
        with param_col:
            num_chunks = st.number_input(
                "Chunks to retrieve:",
                min_value=1,
                max_value=10,
                value=3
            )
        
        with button_col:
            search_clicked = st.button("Search" )
            
        return {
            "query": query,
            "num_chunks": num_chunks,
            "search_triggered": search_clicked
        }

    def render_process_flow(self, current_stage: str = None):
        """Render process flow section (basic version for Phase 1)."""
        st.subheader("Process Flow")
        
        # Simple flow display for Phase 1
        # Will be enhanced with animations and status in Phase 2
        flow_col1, flow_col2, flow_col3, flow_col4 = st.columns(4)
        
        with flow_col1:
            st.text("Query")
        with flow_col2:
            st.text("Vector Search")
        with flow_col3:
            st.text("Chunks")
        with flow_col4:
            st.text("Response")

    def render_results_panel(self, chunks: List[Dict] = None, response: str = None):
        """Render results panel with tabs for chunks and response."""
        chunks_tab, response_tab = st.tabs(["Retrieved Chunks", "Final Response"])
        
        with chunks_tab:
            if chunks:
                for i, chunk in enumerate(chunks, 1):
                    with st.expander(f"Chunk {i}", expanded=True):
                        st.text(chunk.get('text', ''))
                        col1, col2 = st.columns(2)
                        with col1:
                            st.text(f"Source: {chunk.get('source', 'Unknown')}")
                        with col2:
                            st.text(f"Score: {chunk.get('score', 0.0):.2f}")
            else:
                st.info("No chunks retrieved yet")

        with response_tab:
            if response:
                st.write(response)
            else:
                st.info("No response generated yet")

    def render_metrics_display(self, metrics: Dict = None):
        """Render metrics display section."""
        st.subheader("Performance Metrics")
        
        if metrics:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Search Time", f"{metrics.get('search_time', 0):.2f}s")
            with col2:
                st.metric("Chunks Retrieved", metrics.get('num_chunks', 0))
            with col3:
                st.metric("Total Time", f"{metrics.get('total_time', 0):.2f}s")
        else:
            st.info("No metrics available yet")

    def run(self):
        """Main UI execution flow."""
        self.setup_page()
        
        # 1. Query Input Section
        input_state = self.render_query_input()
        
        # 2. Process Flow Section (basic for Phase 1)
        self.render_process_flow()
        
        # Create placeholder for dynamic content
        results_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        # Handle search if triggered
        if input_state["search_triggered"] and input_state["query"]:
            with st.spinner("Processing query..."):
                # Placeholder for actual processing
                # Will be integrated with RAG system later
                example_chunks = [
                    {"text": "Sample chunk 1", "source": "Doc A", "score": 0.95},
                    {"text": "Sample chunk 2", "source": "Doc B", "score": 0.85}
                ]
                example_response = "This is a sample response."
                example_metrics = {
                    "search_time": 0.3,
                    "num_chunks": 2,
                    "total_time": 0.8
                }
                
                # 3. Results Panel
                self.render_results_panel(
                    chunks=example_chunks,
                    response=example_response
                )
                
                # 4. Metrics Display
                self.render_metrics_display(example_metrics)

def main():
    ui = RAGQueryUI()
    ui.run()

if __name__ == "__main__":
    main()