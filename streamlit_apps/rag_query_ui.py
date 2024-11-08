import streamlit as st
import sys
import os
from typing import Dict, List, Any
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.singleton_config import ConfigSingleton
from src.paths import get_db_path, get_faiss_path
from src.embedding_generator_factory import EmbeddingGeneratorFactory
from src.query_analyzer import QueryAnalyzer
from src.reranker_factory import ReRankerFactory
from src.generation import Generator
from src.rag_search_client import RAGSearchClient
from src.logging_config import get_logger

class RAGQueryUI:
    def __init__(self):
        """Initialize the RAG Query UI with all required components."""
        self.config = ConfigSingleton()
        self.logger = get_logger('ui')
        
        try:
            # Initialize core components
            self.query_analyzer = QueryAnalyzer()
            self.reranker = ReRankerFactory.create()
            self.search_client = RAGSearchClient()
            self.embedding_generator = EmbeddingGeneratorFactory.create(
                generator_type=self.config.get_pipeline_config().embedding.provider,
                **self.config.get_active_embedding_config().__dict__
            )
            self.generator = Generator()
            
            self.logger.info("RAG Query UI initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG Query UI: {str(e)}")
            raise

    def setup_page(self):
        """Configure the Streamlit page layout."""
        st.set_page_config(
            page_title="RAG Query System",
            layout="wide",
            initial_sidebar_state="expanded"  # Changed to expanded
        )

        # Add logo and controls in sidebar
        with st.sidebar:
            st.image("../src/static/logo.png", width=100)
            st.markdown("---")
            
            # Configuration options
            st.subheader("Search Settings")
            show_analysis = st.checkbox("Show Query Analysis", value=True)  # Default to True
            show_reranking = st.checkbox("Show Re-ranking Impact", value=True)  # Default to True
            context_only = st.checkbox("Context Only Mode", value=False)
            
            st.markdown("---")
            st.markdown("### About")
            st.markdown("RAG Query System v1.0")

        return {
            "show_analysis": show_analysis,
            "show_reranking": show_reranking,
            "context_only": context_only
        }

###############################
    def render_query_input(self, settings: Dict[str, bool]) -> Dict:
            """Render the query input section."""
            st.title("RAG Query System")
            
            query_col, param_col, button_col = st.columns([3, 1, 1])
            
            with query_col:
                query = st.text_input("Enter your query:", key="main_query")
                
                # Show query analysis if enabled
                if query and settings["show_analysis"]:
                    with st.expander("Query Analysis", expanded=True):
                        analysis = self.query_analyzer.analyze_query(query)
                        st.info(f"Query Complexity Score: {analysis.complexity_score:.2f}")
                        
                        # Display explanation
                        for exp in analysis.explanation:
                            st.write(f"- {exp}")
                        
                        # Display features with normalized progress bars
                        st.write("Feature Breakdown:")
                        max_feature_value = max(analysis.features.values())
                        for feature, value in analysis.features.items():
                            # Normalize to 0-1 range
                            normalized_value = value / (max_feature_value * 1.2)  # Add 20% headroom
                            st.progress(min(normalized_value, 1.0), text=f"{feature}: {value:.2f}")
            
            with param_col:
                num_chunks = st.number_input(
                    "Results to show:",
                    min_value=1,
                    max_value=10,
                    value=3
                )
            
            with button_col:
                search_clicked = st.button("Search")
            
            return {
                "query": query,
                "num_chunks": num_chunks,
                "search_triggered": search_clicked
            }
###############################

    def render_process_flow(self, stage: str = None):
        """Render process flow section."""
        st.subheader("Process Flow")
        
        # Simple flow display
        flow_col1, flow_col2, flow_col3, flow_col4 = st.columns(4)
        
        with flow_col1:
            st.write("Query")
            if stage == "query":
                st.markdown("🔄")
            elif stage and stage > "query":
                st.markdown("✅")
                
        with flow_col2:
            st.write("Vector Search")
            if stage == "search":
                st.markdown("🔄")
            elif stage and stage > "search":
                st.markdown("✅")
                
        with flow_col3:
            st.write("Chunks")
            if stage == "chunks":
                st.markdown("🔄")
            elif stage and stage > "chunks":
                st.markdown("✅")
                
        with flow_col4:
            st.write("Response")
            if stage == "response":
                st.markdown("🔄")
            elif stage and stage > "response":
                st.markdown("✅")

    def _display_reranking_impact(
            self,
            initial_results: List[Dict],
            reranked_results: List
        ):
            """Display a visualization of re-ranking impact."""
            st.subheader("Re-ranking Impact Analysis")
            
            # Create a before/after comparison
            before_after = {
                'Position': list(range(1, len(initial_results) + 1)),
                'Before Re-ranking': [r['relevance_score'] for r in initial_results],
                'After Re-ranking': [r.reranked_score for r in reranked_results]
            }
            
            df = pd.DataFrame(before_after)
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Score Distribution")
                st.bar_chart(df.melt('Position', var_name='Stage', value_name='Score'))
                
            with col2:
                st.write("Position Changes")
                # Create a mapping of chunk_ids to their positions in both lists
                initial_positions = {chunk['chunk_id']: i for i, chunk in enumerate(initial_results)}
                reranked_positions = {r.chunk_id: i for i, r in enumerate(reranked_results)}
                
                # Show position changes
                for chunk_id in initial_positions:
                    if chunk_id in reranked_positions:
                        old_pos = initial_positions[chunk_id] + 1  # 1-based position
                        new_pos = reranked_positions[chunk_id] + 1  # 1-based position
                        if old_pos != new_pos:
                            st.write(f"🔄 Chunk moved: Position {old_pos} → {new_pos}")


    def display_results(
        self,
        results: Dict[str, Any],
        settings: Dict[str, bool]
    ):
        """Display search results and generated response with search metrics."""
        if not results:
            return

        # Display processing information
        st.info(f"Processing time: {results['processing_time']:.2f} seconds")

        # Add Search Strategy and Metrics section
        with st.expander("🔍 Search Analysis", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Search Strategy")
                analysis = results['analysis']
                
                # Show whether re-ranking was used
                if analysis.needs_reranking:
                    st.success("🔄 Re-ranking was applied")
                    st.write(f"- Complexity Score: {analysis.complexity_score:.2f}")
                    st.write(f"- Initial candidates retrieved: {analysis.recommended_candidates}")
                    st.write(f"- Final results shown: {len(results['results'])}")
                else:
                    st.info("📍 Vector search only")
                    st.write(f"- Complexity Score: {analysis.complexity_score:.2f}")
                    st.write("- Re-ranking not needed for this query")
            
            with col2:
                st.subheader("Relevance Metrics")
                if analysis.needs_reranking:
                    # Show before/after stats
                    scores_before = [r['initial_score'] for r in results['results']]
                    scores_after = [r['relevance_score'] for r in results['results']]
                    
                    metrics_df = pd.DataFrame({
                        'Metric': ['Average Score', 'Highest Score', 'Lowest Score'],
                        'Before Re-ranking': [
                            f"{sum(scores_before) / len(scores_before):.3f}",
                            f"{max(scores_before):.3f}",
                            f"{min(scores_before):.3f}"
                        ],
                        'After Re-ranking': [
                            f"{sum(scores_after) / len(scores_after):.3f}",
                            f"{max(scores_after):.3f}",
                            f"{min(scores_after):.3f}"
                        ]
                    })
                    st.table(metrics_df)
                else:
                    # Show vector search stats only
                    scores = [r['relevance_score'] for r in results['results']]
                    st.write(f"- Average relevance: {sum(scores) / len(scores):.3f}")
                    st.write(f"- Highest score: {max(scores):.3f}")
                    st.write(f"- Lowest score: {min(scores):.3f}")
                    st.write(f"- Score spread: {max(scores) - min(scores):.3f}")

        # Tabs for results
        chunks_tab, response_tab = st.tabs(["Retrieved Chunks", "Generated Response"])
        
        with chunks_tab:
            # Display chunks with enhanced score information
            for i, chunk in enumerate(results['results'], 1):
                score_display = f"{chunk['relevance_score']:.3f}"
                if analysis.needs_reranking:
                    score_display += f" (Initial: {chunk.get('initial_score', 0):.3f})"
                    
                with st.expander(
                    f"Chunk {i} (Score: {score_display})",
                    expanded=(i == 1)
                ):
                    st.write(chunk['chunk_text'])
                    if chunk.get('metadata'):
                        st.markdown("---")
                        st.write("Source Information:")
                        for key, value in chunk['metadata'].items():
                            st.write(f"- {key}: {value}")
        
        with response_tab:
            if not settings["context_only"] and results.get('response'):
                st.write(results['response'])
            else:
                st.info("Context-only mode enabled - no response generated")

    def process_query(self, query: str, num_chunks: int, settings: Dict[str, bool]):
        """Process the query and return results."""
        try:
            self.render_process_flow("query")
            start_time = time.time()
            
            # Analyze query
            analysis = self.query_analyzer.analyze_query(query)
            
            # Generate query embedding
            self.render_process_flow("search")
            query_vector = self.embedding_generator.generate_embedding(query)
            
            # Get initial results
            initial_k = analysis.recommended_candidates if analysis.needs_reranking else num_chunks
            initial_results = self.search_client.search(
                query_vector=query_vector,
                k=initial_k
            )
            
            self.render_process_flow("chunks")
            
            # Apply re-ranking if needed
            if analysis.needs_reranking:
                reranked_results = self.reranker.rerank(
                    query=query,
                    candidates=initial_results,
                    top_k=num_chunks
                )
                
                if settings["show_reranking"]:
                    self._display_reranking_impact(
                        initial_results[:num_chunks],
                        reranked_results
                    )
                
                final_results = [
                    {
                        'chunk_id': r.chunk_id,
                        'chunk_text': r.chunk_text,
                        'relevance_score': r.reranked_score,
                        'metadata': r.metadata,
                        'initial_score': r.initial_score
                    }
                    for r in reranked_results
                ]
            else:
                final_results = initial_results[:num_chunks]
            
            self.render_process_flow("response")
            
            # Generate response if not in context-only mode
            if not settings["context_only"]:
                response = self.generator.generate_response(
                    query=query,
                    search_results=final_results,
                    allow_training_data=False
                )
            else:
                response = None
            
            processing_time = time.time() - start_time
            
            self.render_process_flow("complete")
            
            return {
                'results': final_results,
                'response': response,
                'processing_time': processing_time,
                'analysis': analysis
            }
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
            return None

    def run(self):
        """Main execution flow for the UI."""
        # Setup page and get settings
        settings = self.setup_page()
        
        # Get query input
        input_state = self.render_query_input(settings)
        
        # Process query if triggered
        if input_state["search_triggered"] and input_state["query"]:
            with st.spinner("Processing query..."):
                results = self.process_query(
                    input_state["query"],
                    input_state["num_chunks"],
                    settings
                )
                
                if results:
                    self.display_results(results, settings)

def main():
    ui = RAGQueryUI()
    ui.run()

if __name__ == "__main__":
    main()