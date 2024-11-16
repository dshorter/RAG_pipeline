import streamlit as st
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
import time
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Project imports    
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.singleton_config import ConfigSingleton
from src.paths import get_db_path, get_faiss_path
from src.embedding_generator_factory import EmbeddingGeneratorFactory
from src.query_analyzer import QueryAnalyzer
from src.reranker_factory import ReRankerFactory
from src.reranker_base import ReRankerBase
from src.generation import Generator
from src.rag_search_client import RAGSearchClient
from src.logging_config import get_logger, setup_rag_logging

class RAGQueryUI:
    def __init__(self):
        """Initialize the RAG Query UI with all required components."""
        setup_rag_logging(log_dir='logs', unified_log=True)
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
            
            self.logger.info("RAG Query UI initialized successfully", 
                           extra={'component': 'ui', 'operation': 'initialization'})
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG Query UI: {str(e)}", 
                            extra={'component': 'ui', 'operation': 'initialization_error'})
            raise

    def setup_page(self):
        """Configure the Streamlit page layout."""
        st.set_page_config(
            page_title="RAG Query System",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        with st.sidebar:
            st.image("../src/static/logo.png", width=100)
            st.markdown("---")
            
            # Configuration options
            st.subheader("Search Settings")
            show_analysis = st.checkbox("Show Query Analysis", value=True)
            show_reranking = st.checkbox("Show Re-ranking Impact", value=True)
            use_training_data = st.checkbox(
                "Allow Training Data",
                value=False,
                help="When enabled, allows the AI to supplement context with its training data"
            )
            
            st.markdown("---")
            st.markdown("### About")
            st.markdown("RAG Query System v1.0")

        return {
            "show_analysis": show_analysis,
            "show_reranking": show_reranking,
            "use_training_data": use_training_data
        }

    def render_query_input(self, settings: Dict[str, bool]) -> Dict:
        """Render the query input section."""
        st.title("RAG Query System")
        
        query_col, param_col, button_col = st.columns([3, 1, 1])
        
        with query_col:
            query = st.text_input("Enter your query:", key="main_query")
            
            if query and settings["show_analysis"]:
                with st.expander("Query Analysis", expanded=True):
                    analysis = self.query_analyzer.analyze_query(query)
                    st.info(f"Query Complexity Score: {analysis.complexity_score:.2f}")
                    
                    for exp in analysis.explanation:
                        st.write(f"- {exp}")
                    
                    st.write("Feature Breakdown:")
                    max_feature_value = max(analysis.features.values())
                    for feature, value in analysis.features.items():
                        normalized_value = value / (max_feature_value * 1.2)
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

    def render_process_flow(self, stage: str = None):
        """Render process flow section."""
        st.subheader("Process Flow")
        
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

    def _display_reranking_impact(self, initial_results: List[Dict], reranked_results: List):
        """Display a visualization of re-ranking impact."""
        try:
            st.subheader("Re-ranking Impact Analysis")
            
            chart_data = pd.DataFrame({
                'Vector Search': [r['relevance_score'] for r in initial_results],
                'After Re-ranking': [r.reranked_score for r in reranked_results]
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Score Distribution by Position")
                try:
                    st.bar_chart(chart_data)
                    
                    st.markdown("""
                    **Legend:**
                    - Dark Blue: Original Vector Search Scores
                    - Light Blue: Scores After Re-ranking
                    """)
                    
                    st.write("Position Reference:")
                    for i in range(len(initial_results)):
                        st.write(f"Position {i+1}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to render chart: {str(e)}")
                    st.write("Scores:", chart_data.to_dict())
            
            with col2:
                st.write("Position Changes")
                st.write("📊 What Changed:")
                
                for i, (before, after) in enumerate(zip(initial_results, reranked_results)):
                    if before['chunk_id'] != after.chunk_id:
                        old_pos = i + 1
                        new_pos = next(j + 1 for j, r in enumerate(reranked_results) 
                                    if r.chunk_id == before['chunk_id'])
                        
                        score_change = after.reranked_score - before['relevance_score']
                        change_icon = "🔺" if score_change > 0 else "🔻"
                        
                        st.markdown(f"""
                            **Chunk {before['chunk_id']}**:
                            - Moved from position {old_pos} to {new_pos}
                            - Score change: {change_icon} {abs(score_change):.3f}
                            - Relevance impact: {'Improved' if score_change > 0 else 'Decreased'}
                        """)

                if all(before['chunk_id'] == after.chunk_id 
                      for before, after in zip(initial_results, reranked_results)):
                    st.info("No significant position changes after re-ranking.")
                
                st.markdown("""
                #### Understanding Re-ranking:
                - Position changes indicate revised relevance assessment
                - 🔺 indicates improved relevance score
                - 🔻 indicates decreased relevance score
                - Stable positions suggest consistent relevance
                """)
                
        except Exception as e:
            self.logger.error(f"Error in _display_reranking_impact: {str(e)}")
            st.error("Failed to display re-ranking impact visualization. Results are still valid.")

    def _display_citation_details(self, citation_data: Dict[str, Any], citation_num: str):
        """Display detailed citation information in an expander."""
        with st.expander(f"Source {citation_num}", expanded=True):
            source_info = citation_data['source_info']
            st.markdown("#### Source Information")
            st.write(f"**Title:** {source_info.get('title', 'Unknown')}")
            st.write(f"**Author:** {source_info.get('author', 'Unknown')}")
            if 'source' in source_info:
                st.write(f"**Source:** {source_info['source']}")
            
            st.markdown("#### Reference Details")
            if 'start_index' in source_info and 'end_index' in source_info:
                st.write(f"**Section:** {source_info['start_index']}-{source_info['end_index']}")
            st.write(f"**Relevance Score:** {citation_data['relevance_score']:.2f}")
            
            st.markdown("#### Text Preview")
            preview_text = citation_data['chunk_text'][:200]
            if len(citation_data['chunk_text']) > 200:
                preview_text += "..."
            st.text(preview_text)

    def display_response_with_citations(self, response_data: Dict[str, Any], 
                                 settings: Dict[str, bool]):
        """Display response with simple footnotes."""
        if settings["use_training_data"]:
            st.info("🔄 Response may include training data to supplement context")
        else:
            st.info("📚 Response based solely on provided context")

        # Simply display the full response text which includes the footnotes
        st.markdown(response_data["response_text"])
    
    def display_results(self, results: Dict[str, Any], settings: Dict[str, bool]):
        """Display search results and generated response with search metrics."""
        if not results:
            return

        st.info(f"Processing time: {results['processing_time']:.2f} seconds")

        with st.expander("🔍 Search Analysis", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Search Strategy")
                analysis = results['analysis']
                
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
                    scores_before = [r.get('initial_score', r.get('relevance_score', 0)) for r in results['results']]
                    scores_after = [r.get('relevance_score', 0) for r in results['results']]
                    
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
                    scores = [r.get('relevance_score', 0) for r in results['results']]
                    st.write(f"- Average relevance: {sum(scores) / len(scores):.3f}")
                    st.write(f"- Highest score: {max(scores):.3f}")
                    st.write(f"- Lowest score: {min(scores):.3f}")
                    st.write(f"- Score spread: {max(scores) - min(scores):.3f}")

        chunks_tab, response_tab = st.tabs(["Retrieved Chunks", "Generated Response"])
        
        with chunks_tab:
            for i, chunk in enumerate(results['results'], 1):
                score_display = f"{chunk['relevance_score']:.3f}"
                if results['analysis'].needs_reranking:
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
            if results['response'] and 'response_text' in results['response']:
                if settings["use_training_data"]:
                    st.info("🤖 Response may include information from AI training data")
                else:
                    st.info("📚 Response based strictly on provided context")
                
                self.display_response_with_citations(results['response'], settings)


    def process_query(self, query: str, num_chunks: int, settings: Dict[str, bool]):
        """Process the query and return results with citations."""
        try:
            self.render_process_flow("query")
            start_time = time.time()
            
            # Query analysis
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
                
                # Preserve all metadata when creating final results
                final_results = []
                for r in reranked_results:
                    # Find the original result to get its metadata
                    original_result = next(
                        res for res in initial_results 
                        if res['chunk_id'] == r.chunk_id
                    )
                    
                    final_results.append({
                        'chunk_id': r.chunk_id,
                        'chunk_text': r.chunk_text,
                        'document_id': original_result.get('document_id'),
                        'relevance_score': r.reranked_score,
                        'initial_score': r.initial_score,
                        'metadata': original_result.get('metadata', {}),
                        'source_info': original_result.get('source_info', {}),
                        'document_metadata': original_result.get('document_metadata', {})
                    })
            else:
                final_results = initial_results[:num_chunks]
            
            self.render_process_flow("response")
            
            # Generate response with citations
            response_data = self.generator.generate_response(
                query=query,
                search_results=final_results,
                allow_training_data=settings["use_training_data"]
            )
            
            processing_time = time.time() - start_time
            
            self.render_process_flow("complete")
            
            return {
                'results': final_results,
                'response': response_data,
                'processing_time': processing_time,
                'analysis': analysis
            }
                
        except Exception as e:
            self.logger.error(f"Query processing failed: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
            return None
        
    ##############################################
    
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




    