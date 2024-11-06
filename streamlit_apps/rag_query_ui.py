import streamlit as st
from typing import Dict, List
import time    
import os, sys 

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.singleton_config import ConfigSingleton
from src.paths import get_db_path, get_faiss_path
from src.embedding_generator_factory import EmbeddingGeneratorFactory
from src.generation import Generator
from src.rag_search_client import RAGSearchClient    

class RAGQueryUI:
    def __init__(self):
        self.config = ConfigSingleton()
        # Initialize core components
        self.rag_search_client = RAGSearchClient()
        self.generatorX = Generator()
        self.embedding_generator = EmbeddingGeneratorFactory.create(
            generator_type=self.config.get_pipeline_config().embedding.provider,
            **self.config.get_active_embedding_config().__dict__
        )

    def setup_page(self):
        """Initialize page configuration."""
        st.set_page_config(
            page_title="RAG Query System",
            layout="wide",
            initial_sidebar_state="collapsed"
        )

        # Add custom CSS to align the logo with the title
        st.markdown("""
            <style>
            .stMarkdown svg {
                margin-top: 1rem;
                vertical-align: middle;
            }
            </style>
        """, unsafe_allow_html=True)

    def render_query_input(self) -> Dict:
        """Render query input section with mode selection."""
        # Logo and title
        logo_col, title_col = st.columns([1, 4])
        
        with logo_col:
            st.markdown("""
            <svg width="50" height="50" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
                <style>
                    .node { fill: #6ba4ff; }
                    .central-node { fill: #4285f4; }
                    .line { stroke: #a8c1ff; stroke-width: 2; }
                    .highlight { stroke: #4285f4; stroke-width: 3; }
                </style>
                <line class="line" x1="100" y1="50" x2="50" y2="150" />
                <line class="line" x1="100" y1="50" x2="150" y2="150" />
                <line class="line" x1="50" y1="150" x2="150" y2="150" />
                <line class="highlight" x1="100" y1="50" x2="100" y2="100" />
                <circle class="node" cx="50" cy="150" r="10" />
                <circle class="node" cx="150" cy="150" r="10" />
                <circle class="central-node" cx="100" cy="50" r="15" />
                <g transform="translate(150, 50)">
                    <circle cx="0" cy="0" r="5" fill="none" stroke="#6ba4ff" stroke-width="2"/>
                    <line x1="4" y1="4" x2="10" y2="10" stroke="#6ba4ff" stroke-width="2"/>
                </g>
            </svg>
            """, unsafe_allow_html=True)
        
        with title_col:
            st.title("RAG Query System")
        
        # Mode selection with explanation
        mode_container = st.container()
        with mode_container:
            st.write("### Response Mode")
            mode_col1, mode_col2 = st.columns([3, 1])
            
            with mode_col1:
                allow_training_data = st.checkbox(
                    "Allow AI to supplement with its knowledge",
                    help="When enabled, AI can add relevant information from its training data when context is insufficient"
                )
            
            # Show current mode explanation
            with mode_col2:
                if allow_training_data:
                    st.info("📚 Using context + AI knowledge")
                else:
                    st.info("📄 Using context only")
            
            # Mode description
            if allow_training_data:
                st.markdown("""
                > In this mode, the AI will:
                > - Primarily use information from the provided documents
                > - Supplement with additional knowledge when relevant
                > - Clearly mark any information from its training data
                """)
            else:
                st.markdown("""
                > In this mode, the AI will:
                > - Use only information from the provided documents
                > - Indicate when information is not available in the context
                > - Not supplement with additional knowledge
                """)
        
        st.markdown("---")
        
        # Query input section
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
            search_clicked = st.button("Search")
            
        return {
            "query": query,
            "num_chunks": num_chunks,
            "search_triggered": search_clicked,
            "allow_training_data": allow_training_data
        }
    def render_process_flow(self, active_stage: str = None):
        """Render process flow section with status indicators."""
        st.subheader("Process Flow")
        
        stages = {
            "query": ("Query", "🔵"),
            "vector_search": ("Vector Search", "🔵"),
            "chunks": ("Chunks", "🔵"),
            "response": ("Response", "🔵")
        }
        
        if active_stage:
            stages[active_stage] = (stages[active_stage][0], "🔄")
        
        flow_cols = st.columns(4)
        for i, (stage, (name, icon)) in enumerate(stages.items()):
            with flow_cols[i]:
                st.write(f"{icon} {name}")

    def render_results_panel(self, chunks: List[Dict] = None, response: str = None):
        """Render results panel with chunks and response."""
        chunks_tab, response_tab = st.tabs(["Retrieved Chunks", "Final Response"])
        
        with chunks_tab:
            if chunks:
                for i, chunk in enumerate(chunks, 1):
                    with st.expander(f"Chunk {i}", expanded=True):
                        st.write(chunk['chunk_text'])
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"Source: {chunk.get('source_info', {}).get('title', 'Unknown')}")
                            st.write(f"Author: {chunk.get('source_info', {}).get('author', 'Unknown')}")
                        with col2:
                            st.write(f"Relevance Score: {chunk.get('relevance_score', 0.0):.3f}")
                            if 'citation' in chunk:
                                st.write("Citation:", chunk['citation'])
            else:
                st.info("No chunks retrieved yet")

        with response_tab:
            if response:
                st.markdown(response)
                st.divider()
                st.caption("""
                Note: When enabled, information from AI's training data is marked with [AI Knowledge: ...].
                All other information comes directly from the provided documents.
                """)
            else:
                st.info("No response generated yet")

    def render_metrics_display(self, metrics: Dict = None):
        """Render metrics display section."""
        st.subheader("Performance Metrics")
        
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Embedding Time", f"{metrics.get('embedding_time', 0):.3f}s")
            with col2:
                st.metric("Search Time", f"{metrics.get('search_time', 0):.3f}s")
            with col3:
                st.metric("Response Time", f"{metrics.get('response_time', 0):.3f}s")
            with col4:
                st.metric("Total Time", f"{metrics.get('total_time', 0):.3f}s")

            st.metric("Chunks Retrieved", metrics.get('num_chunks', 0))
        else:
            st.info("No metrics available yet")

    def process_query(self, query: str, num_chunks: int, allow_training_data: bool = False) -> Dict:
        """Process query through the RAG pipeline."""
        try:
            results = {}
            
            # Generate query embedding
            with st.spinner("Generating query embedding..."):
                start_time = time.time()
                query_vector = self.embedding_generator.generate_embedding(query)
                embedding_time = time.time() - start_time
                results['embedding_time'] = embedding_time

            # Search for relevant chunks
            with st.spinner("Searching for relevant chunks..."):
                start_time = time.time()
                search_results = self.rag_search_client.search(query_vector, k=num_chunks)
                search_time = time.time() - start_time
                results['search_time'] = search_time
                results['chunks'] = search_results

            # Generate response using chunks
            with st.spinner("Generating response..."):
                start_time = time.time()
                generated_response = self.generatorX.generate_response(
                    query, 
                    search_results,
                    allow_training_data=allow_training_data
                )
                response_time = time.time() - start_time
                results['response_time'] = response_time
                results['response'] = generated_response

            # Collect metrics
            results['total_time'] = embedding_time + search_time + response_time
            results['num_chunks'] = len(search_results)

            return results

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            raise

    def run(self):
        """Main UI execution flow."""
        self.setup_page()
        
        # 1. Query Input Section with mode selection
        input_state = self.render_query_input()
        
        # 2. Process Flow Section
        flow_container = st.empty()
        flow_container.container()
        self.render_process_flow()
        
        # Handle search if triggered
        if input_state["search_triggered"] and input_state["query"]:
            try:
                flow_container.container()
                self.render_process_flow("query")
                
                # Execute search with mode selection
                results = self.process_query(
                    input_state["query"],
                    input_state["num_chunks"],
                    input_state["allow_training_data"]
                )
                
                if results:
                    # 3. Results Panel
                    self.render_results_panel(
                        chunks=results.get('chunks'),
                        response=results.get('response')
                    )
                    
                    # 4. Metrics Display
                    self.render_metrics_display(results)
                    
                    # Final flow status
                    flow_container.container()
                    self.render_process_flow("response")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

def main():
    ui = RAGQueryUI()
    ui.run()

if __name__ == "__main__":
    main()