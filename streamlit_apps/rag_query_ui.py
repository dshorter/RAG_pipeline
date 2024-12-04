import os
import sys
import streamlit as st
from typing import Dict, List, Any, Optional
import time
from datetime import datetime
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
from src.metrics_collector import ChatHistoryStore    
from src.logging_config import setup_rag_logging, get_logger


# Set up logging
setup_rag_logging(log_dir='logs', unified_log=True)
logger = get_logger(__name__)

class RAGQueryUI:

#############
    def __init__(self):
        """Initialize the RAG Query UI with session state management."""
        # Initialize session state first thing
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.settings = {
                "show_analysis": True,
                "show_reranking": True,
                "use_training_data": False
            }
            st.session_state.query_history = []
            st.session_state.last_results = None
            st.session_state.current_stage = None
        
        if not st.session_state.initialized:
            try:
                # Core components
                st.session_state.config = ConfigSingleton()
                st.session_state.logger = get_logger('ui')
                st.session_state.query_analyzer = QueryAnalyzer()
                st.session_state.reranker = ReRankerFactory.create()
                st.session_state.search_client = RAGSearchClient()
                st.session_state.embedding_generator = EmbeddingGeneratorFactory.create(
                    generator_type=st.session_state.config.get_pipeline_config().embedding.provider,
                    **st.session_state.config.get_active_embedding_config().__dict__
                )
                st.session_state.generator = Generator()
                
                # Initialize chat store once
                st.session_state.chat_store = ChatHistoryStore()
                logger.info("Chat history store initialized")
                
                # Instance methods/properties
                self.config = st.session_state.config
                self.logger = st.session_state.logger
                self.query_analyzer = st.session_state.query_analyzer
                self.reranker = st.session_state.reranker
                self.search_client = st.session_state.search_client
                self.embedding_generator = st.session_state.embedding_generator
                self.generator = st.session_state.generator
                self.chat_store = st.session_state.chat_store
                
                st.session_state.initialized = True
                logger.info("Session state initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize components: {str(e)}")
                st.error(f"Failed to initialize components: {str(e)}")
                raise
#############
    def process_query(self, query: str, num_chunks: int, settings: Dict[str, bool]):
        """Process the query and store results in session state."""
        try:
            self.render_process_flow("query")
            start_time = time.time()
            
            # Store query in history
            st.session_state.query_history.append((query, datetime.now()))
            
            # Query analysis
            analysis = st.session_state.query_analyzer.analyze_query(query)
            
            # Generate query embedding
            self.render_process_flow("search")
            query_vector = st.session_state.embedding_generator.generate_embedding(query)
            
            # Get initial results
            initial_k = analysis.recommended_candidates if analysis.needs_reranking else num_chunks
            initial_results = st.session_state.search_client.search(
                query_vector=query_vector,
                k=initial_k
            )
            
            self.render_process_flow("chunks")
            
            # Apply re-ranking if needed
            if analysis.needs_reranking:
                reranked_results = st.session_state.reranker.rerank(
                    query=query,
                    candidates=initial_results,
                    top_k=num_chunks
                )
                
                if settings["show_reranking"]:
                    self._display_reranking_impact(
                        initial_results[:num_chunks],
                        reranked_results
                    )
                
                final_results = []
                for r in reranked_results:
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
            response_data = st.session_state.generator.generate_response(
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
#############

    def render_query_input(self, settings: Dict[str, bool]) -> Dict:
        """Render the query input section."""
        st.title("RAG Query System")
        
        query_col, param_col, button_col = st.columns([3, 1, 1])
        
        with query_col:
            query = st.text_input("Enter your query:", key="main_query")
            
            # Show query analysis if enabled
            if query and settings["show_analysis"]:
                with st.expander("Query Analysis", expanded=True):
                    analysis = st.session_state.query_analyzer.analyze_query(query)
                    st.info(f"Query Complexity Score: {analysis.complexity_score:.2f}")
                    
                    # Display explanation
                    for exp in analysis.explanation:
                        st.write(f"- {exp}")
                    
                    # Display features with normalized progress bars
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
        st.session_state.current_stage = stage
        
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


    def _calculate_improvement(self, initial: float, final: float) -> str:
        """
        Calculate and format the percentage improvement between scores.
        """
        try:
            if initial > 0:
                pct_change = ((final - initial) / initial) * 100
                return f"+{pct_change:.0f}%" if pct_change > 0 else f"{pct_change:.0f}%"
            return "N/A"
        except Exception as e:
            self.logger.error(f"Error calculating improvement: {str(e)}")
            return "N/A"

    def display_results(self, results: Dict[str, Any], settings: Dict[str, bool]):
        """Display search results and generated response with search metrics."""
        if not results:
            return

        st.info(f"Processing time: {results['processing_time']:.2f} seconds")

        # Display analysis expander
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
                    scores_before = [r.get('initial_score', r.get('relevance_score', 0)) 
                                for r in results['results']]
                    scores_after = [r.get('relevance_score', 0) for r in results['results']]
                    
                    # Calculate average improvement
                    avg_improvement = self._calculate_improvement(
                        sum(scores_before) / len(scores_before),
                        sum(scores_after) / len(scores_after)
                    )
                    
                    metrics_df = pd.DataFrame({
                        'Metric': ['Average Score', 'Highest Score', 'Lowest Score', 'Avg Improvement'],
                        'Before Re-ranking': [
                            f"{sum(scores_before) / len(scores_before):.3f}",
                            f"{max(scores_before):.3f}",
                            f"{min(scores_before):.3f}",
                            "N/A"
                        ],
                        'After Re-ranking': [
                            f"{sum(scores_after) / len(scores_after):.3f}",
                            f"{max(scores_after):.3f}",
                            f"{min(scores_after):.3f}",
                            avg_improvement
                        ]
                    })
                    st.table(metrics_df)
                else:
                    scores = [r.get('relevance_score', 0) for r in results['results']]
                    st.write(f"- Average relevance: {sum(scores) / len(scores):.3f}")
                    st.write(f"- Highest score: {max(scores):.3f}")
                    st.write(f"- Lowest score: {min(scores):.3f}")
                    st.write(f"- Score spread: {max(scores) - min(scores):.3f}")

        # Create tabs for results
        chunks_tab, response_tab = st.tabs(["Retrieved Chunks", "Generated Response"])
        
        with chunks_tab:
            for i, chunk in enumerate(results['results'], 1):
                # Score and improvement calculation
                current_score = chunk['relevance_score']
                
                if analysis.needs_reranking:
                    initial_score = chunk.get('initial_score', 0)
                    improvement = self._calculate_improvement(initial_score, current_score)
                    
                    # Create the expander header
                    expander_label = (
                        f"Chunk {i} - Score: {current_score:.3f} "
                        f"(Initial: {initial_score:.3f}, "
                        f"Improvement: {improvement})"
                    )
                else:
                    expander_label = f"Chunk {i} - Score: {current_score:.3f}"

                with st.expander(expander_label, expanded=(i == 1)):
                    st.write(chunk['chunk_text'])
                    
                    # Show score improvement metrics if re-ranked
                    if analysis.needs_reranking:
                        score_col1, score_col2, score_col3 = st.columns(3)
                        with score_col1:
                            st.metric("Initial Score", f"{initial_score:.3f}")
                        with score_col2:
                            st.metric("Final Score", f"{current_score:.3f}")
                        with score_col3:
                            st.markdown(
                                f"<h3 style='color: {'green' if current_score > initial_score else 'red'};'>"
                                f"Improvement: {improvement}</h3>",
                                unsafe_allow_html=True
                            )
                    
                    # Show metadata if available
                    if chunk.get('metadata'):
                        st.markdown("---")
                        st.write("Source Information:")
                        for key, value in chunk['metadata'].items():
                            st.write(f"- {key}: {value}")

        with response_tab:
            if results.get('response'):
                if settings["use_training_data"]:
                    st.info("🤖 Response may include information from AI training data")
                else:
                    st.info("📚 Response based strictly on provided context")
                st.markdown(results['response']["response_text"])
        def display_response_with_citations(self, response_data: Dict[str, Any], 
                                        settings: Dict[str, bool]):
            """Display response with citations."""
            if settings["use_training_data"]:
                st.info("🔄 Response may include training data to supplement context")
            else:
                st.info("📚 Response based solely on provided context")

            st.markdown(response_data["response_text"])


    def setup_page(self) -> Dict[str, bool]:
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
            
            # Update settings using current values
            current_settings = {
                "show_analysis": st.checkbox(
                    "Show Query Analysis",
                    value=st.session_state.settings["show_analysis"]
                ),
                "show_reranking": st.checkbox(
                    "Show Re-ranking Impact",
                    value=st.session_state.settings["show_reranking"]
                ),
                "use_training_data": st.checkbox(
                    "Allow Training Data",
                    value=st.session_state.settings["use_training_data"]
                )
            }
            st.session_state.settings.update(current_settings)
            
            # History
            if st.checkbox("Show Chat History"):
                st.markdown("### Recent Interactions")
                if hasattr(st.session_state, 'chat_store'):
                    history = st.session_state.chat_store.get_metrics_summary()
                    st.write(f"Total Queries: {history['total_queries']}")
                    st.write(f"Avg Processing Time: {history['avg_processing_time']:.2f}s")
                    
                    recent = st.session_state.chat_store.get_recent_history(5)
                    for interaction in recent:
                        with st.expander(f"{interaction['timestamp']}: {interaction['query'][:30]}..."):
                            st.write(f"Processing Time: {interaction['processing_time']:.2f}s")
                            st.write(f"Chunks Used: {interaction['num_chunks']}")
            
            st.markdown("---")
            st.markdown("### About")
            st.markdown("RAG Query System v1.0")
            
            if st.button("Clear Session"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.experimental_rerun()

        return st.session_state.settings


    def run(self):
        """Main execution flow for the UI."""
        settings = self.setup_page()
        
        # Only proceed if initialization was successful
        if st.session_state.initialized:
            input_state = self.render_query_input(settings)
            
            if input_state["search_triggered"] and input_state["query"]:
                with st.spinner("Processing query..."):
                    results = self.process_query(  # This is the missing reference
                        input_state["query"],
                        input_state["num_chunks"],
                        settings
                    )
                    
                    if results:
                        # Log interaction
                        st.session_state.chat_store.log_interaction(
                            query=input_state["query"],
                            response=results['response'],
                            results=results,
                            processing_time=results['processing_time']
                        )
                        
                        self.display_results(results, settings)
            elif hasattr(st.session_state, 'last_results') and st.session_state.last_results:
                self.display_results(st.session_state.last_results, settings)
        else:
            st.error("System not properly initialized. Please refresh the page.")


    def _display_reranking_impact(
        self,
        initial_results: List[Dict],
        reranked_results: List
    ):
        """Display a visualization of re-ranking impact."""
        st.subheader("Re-ranking Impact Analysis")
        
        try:
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
                    st.session_state.logger.error(f"Failed to render chart: {str(e)}")
                    st.write("Scores:", chart_data.to_dict())
            
            with col2:
                st.write("Position Changes")
                st.write("📊 What Changed:")
                
                for i, (before, after) in enumerate(zip(initial_results, reranked_results)):
                    if before['chunk_id'] != after.chunk_id:
                        old_pos = i + 1
                        try:
                            # Add safeguard for chunk lookup
                            new_pos = next((j + 1 for j, r in enumerate(reranked_results) 
                                        if r.chunk_id == before['chunk_id']), None)
                            
                            if new_pos is not None:  # Only show if found
                                score_change = after.reranked_score - before['relevance_score']
                                change_icon = "🔺" if score_change > 0 else "🔻"
                                st.markdown(f"""
                                    **Chunk {before['chunk_id']}**:
                                    - Moved from position {old_pos} to {new_pos}
                                    - Score change: {change_icon} {abs(score_change):.3f}
                                """)
                        except AttributeError:  # Handle potential missing attributes
                            logger.warning(f"Missing attributes in results for chunk {before['chunk_id']}")
                            continue
                            
        except Exception as e:
            st.session_state.logger.error(f"Error in _display_reranking_impact: ----> {str(e)}\nFull error: {repr(e)}", 
                    exc_info=True)  
            st.error( f"Failed to display re-ranking impact visualization. Results are still valid. {e }  ")

def main():
    ui = RAGQueryUI()
    ui.run()

if __name__ == "__main__":
    main()