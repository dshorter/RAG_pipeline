"""
RAG Chat Interface Module - A Streamlit-based chat interface for the RAG system.

This module implements a chat-style interface for interacting with the RAG system,
featuring custom UI elements, citation handling, and a professional appearance.
It preserves all header, footer, and status elements while ensuring proper
citation display for each response.
"""

import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
import sys
import os
from pathlib import Path

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.singleton_config import ConfigSingleton
from src.embedding_generator_factory import EmbeddingGeneratorFactory
from src.query_analyzer import QueryAnalyzer
from src.reranker_factory import ReRankerFactory
from src.generation import Generator
from src.rag_search_client import RAGSearchClient
from src.logging_config import get_logger
from src.metrics_collector import MetricsCollector

logger = get_logger('chat')

class RAGChatInterface:
    """Chat interface for interacting with the RAG system."""
    
    def __init__(self):
        """Initialize the chat interface with all required components."""
        # Initialize chat history in session state
        if 'messages' not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                    "timestamp": datetime.now().strftime("%H:%M")
                }
            ]
        
        # Initialize RAG components
        if 'interface' not in st.session_state:
            st.session_state.interface = self._initialize_components()
            
        self.metrics_collector = MetricsCollector()

    def _initialize_components(self) -> Dict:
        """Initialize all required RAG system components."""
        try:
            config = ConfigSingleton()
            embedding_config = config.get_active_embedding_config()
            
            return {
                'config': config,
                'query_analyzer': QueryAnalyzer(),
                'reranker': ReRankerFactory.create(),
                'search_client': RAGSearchClient(),
                'generator': Generator(),
                'embedding_generator': EmbeddingGeneratorFactory.create(
                    generator_type=config.get_pipeline_config().embedding.provider,
                    endpoint=embedding_config
                )
            }
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            st.error("Failed to initialize chat components. Please refresh the page.")
            raise

    def setup_page(self):
        """Configure the Streamlit page layout with professional styling."""
        st.set_page_config(
            page_title="ChatDRSC - Research Query Interface",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Add custom CSS for professional appearance
        st.markdown("""
        <style>
        .stApp {
            background-color: #ffffff;
            min-height: 100vh;
            padding-bottom: 60px;
        }
        .main > div {
            padding-top: 0 !important;
            margin-top: -2rem;
        }
        .block-container {
            padding-top: 0;
            max-width: none;
        }
        
        /* Hide default elements */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Professional chat styling */
        .stChatMessage {
            color: #1e293b !important;
            font-size: 18px !important;
        }
        
        /* FIX: Force consistent font sizes for ALL text in chat messages */
        .stChatMessage div[data-testid="stMarkdownContainer"] * {
            font-size: 18px !important;
            line-height: 1.6 !important;
            color: #1e293b !important;  /* Ensure text color is visible */
        }
        
        /* FIX: Specifically target paragraphs that might be rendered as larger text */
        .stChatMessage div[data-testid="stMarkdownContainer"] p {
            font-size: 18px !important;
            margin-bottom: 1rem !important;
            color: #1e293b !important;  /* Dark slate blue color for text */
        }
        
        /* FIX: Target summary sections and AI knowledge blocks which may use different formatting */
        .stChatMessage div[data-testid="stMarkdownContainer"] blockquote,
        .stChatMessage div[data-testid="stMarkdownContainer"] h1,
        .stChatMessage div[data-testid="stMarkdownContainer"] h2,
        .stChatMessage div[data-testid="stMarkdownContainer"] h3,
        .stChatMessage div[data-testid="stMarkdownContainer"] h4 {
            font-size: 18px !important;
            font-weight: normal !important;
            margin: 1rem 0 !important;
            padding: 0 !important;
            color: #1e293b !important;  /* Ensure heading text is visible */
        }

        /* Citation styling */
        .citation-container {
            margin-top: 1rem;
            padding: 1rem;
            background-color: #f8fafc;
            border-radius: 0.5rem;
            border-left: 4px solid #3b82f6;
            font-size: 0.9rem !important;
            color: #475569 !important;
            line-height: 1.5;
        }
        
        /* FIX: Make sure citations have consistent font size but DIFFERENT color */
        .citation-container * {
            font-size: 0.9rem !important;
            color: #475569 !important;  /* Lighter color for citations */
        }

        /* UI Elements */
        .drsc-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: rgba(248, 250, 252, 0.85);
            backdrop-filter: blur(8px);
            border-top: 1px solid #e2e8f0;
            z-index: 1000;
        }
        
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: #22c55e;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        </style>
    """, unsafe_allow_html=True)
        
        # Add professional header with experimental badge
        st.markdown("""
            <div style="background-color: #0f172a; color: white; padding: 1.5rem 2rem; margin: -4rem -4rem 0 -4rem;">
                <div style="max-width: 1200px; margin: 0 auto;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <div style="display: flex; align-items: center;">
                            <h1 style="margin: 0; font-size: 1.5rem;">ChatDRSC</h1>
                            <span style="margin-left: 8px; background-color: #eab308; color: black; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 500;">
                                EXPERIMENTAL
                            </span>
                        </div>
                        <div style="font-size: 0.875rem; color: #94a3b8;">Division of Regulatory Science and Compliance</div>
                    </div>
                    <div style="font-size: 0.875rem; color: #94a3b8;">
                        Research Knowledge Query System
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Add beta notice
        st.markdown("""
            <div style="background-color: rgba(254, 243, 199, 0.3); border-color: #f59e0b; padding: 1rem; border-radius: 0.375rem;">
                <div style="color: #92400e;">
                    🔬 Research Preview: This experimental query interface provides AI-assisted access to research knowledge bases. 
                    Results require validation against primary sources.
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Add footer with system status
        st.markdown("""
            <div class="drsc-footer">
                <div style="max-width: 1200px; margin: 0 auto; padding: 0.75rem; display: flex; justify-content: space-between; align-items: center; font-size: 0.875rem;">
                    <div style="display: flex; gap: 2rem; color: #475569;">
                        <span>DRSC Knowledge Query System</span>
                        <span>Build 0.1.231210</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 2rem;">
                        <div>
                            <span class="status-dot"></span>
                            <span style="color: #475569;">System Status: Research Preview</span>
                        </div>
                        <div style="border-left: 1px solid #e2e8f0; padding-left: 2rem; color: #475569;">
                            Last Updated: Dec 2024
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    def setup_sidebar(self) -> Dict[str, bool]:
        """Configure chat settings in sidebar."""
        with st.sidebar:
            st.title("Chat Settings")
            
            settings = {
                "context_size": st.slider("Context Size", 2, 10, 5),
                "use_training_data": st.toggle("Allow Training Data", False),
                "show_sources": st.toggle("Show Sources", True),
                "show_analysis": st.toggle("Show Query Analysis", False)
            }
            
            st.divider()
            if st.button("Clear Chat History"):
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": "Chat history cleared. How can I help?",
                        "timestamp": datetime.now().strftime("%H:%M")
                    }
                ]
                st.rerun()
            
            st.divider()
            st.caption("Chat Statistics")
            user_messages = sum(1 for msg in st.session_state.messages if msg["role"] == "user")
            st.text(f"Messages: {len(st.session_state.messages)}")
            st.text(f"Questions Asked: {user_messages}")
            
            return settings

    def format_sources(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into readable citations."""
        if not results:
            return ""

        sources = []
        for i, result in enumerate(results, 1):
            source_info = result.get('source_info', {})
            source_parts = []
            
            # Build comprehensive citation
            if title := source_info.get('title'):
                source_parts.append(f"'{title}'")
            if author := source_info.get('author'):
                source_parts.append(f"by {author}")
            if source := source_info.get('source'):
                source_parts.append(f"from {source}")
            
            # Add formatted citation with score
            citation = f"[{i}] {' '.join(source_parts)}"
            if 'relevance_score' in result:
                citation += f" (Relevance: {result['relevance_score']:.2f})"
            
            sources.append(citation)

        return "References:\n" + "\n".join(sources)

    def display_message(self, message: Dict[str, Any], settings: Dict[str, bool]):
        """Display a chat message with its citations."""
        with st.chat_message(message["role"]):
            # Display main message content
            st.markdown(message["content"])
            
            # Display sources if available and enabled
            if settings["show_sources"] and "sources" in message and message["sources"]:
                with st.expander("📚 View Sources", expanded=False):
                    formatted_sources = message["sources"].replace('[', '<b>[').replace(']', ']</b>').replace('\n', '<br>')
                st.markdown(
                    '<div class="citation-container">'
                    f'{formatted_sources}'
                    '</div>',
                    unsafe_allow_html=True
                )
                
            # Show timestamp
            if "timestamp" in message:
                st.caption(f"Sent at {message['timestamp']}")

    def process_query(self, query: str, settings: Dict[str, bool]) -> Dict[str, Any]:
        """Process a user query and generate a response with citations."""
        try:
            start_time = time.time()
            interface = st.session_state.interface
            
            # Analyze query and get embeddings
            analysis = interface['query_analyzer'].analyze_query(query)
            query_vector = interface['embedding_generator'].generate_embedding(query)
            
            # Get initial search results
            initial_k = analysis.recommended_candidates if analysis.needs_reranking else settings['context_size']
            initial_results = interface['search_client'].search(
                query_vector=query_vector,
                k=initial_k
            )
            
            # Apply reranking if needed
            if analysis.needs_reranking:
                reranked_results = interface['reranker'].rerank(
                    query=query,
                    candidates=initial_results,
                    top_k=settings['context_size']
                )
                final_results = [{
                    'chunk_id': r.chunk_id,
                    'chunk_text': r.chunk_text,
                    'relevance_score': r.reranked_score,
                    'initial_score': r.initial_score,
                    'metadata': r.metadata,
                    'source_info': next((res['source_info'] for res in initial_results 
                                   if res['chunk_id'] == r.chunk_id), {})
                } for r in reranked_results]
            else:
                final_results = initial_results[:settings['context_size']]
            
            # Generate response
            response = interface['generator'].generate_response(
                query=query,
                search_results=final_results,
                allow_training_data=True  #settings['use_training_data']
            )

            # Log metrics
            self.metrics_collector.collect(
                operation='chat_query',
                component='chat_interface',
                metrics={
                    'query_length': len(query),
                    'processing_time': time.time() - start_time,
                    'reranking_applied': analysis.needs_reranking,
                    'num_results': len(final_results),
                    'training_data_used': settings['use_training_data']
                }
            )
            
            return {
                'response': response,
                'results': final_results,
                'analysis': analysis,
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {'error': str(e)}

 
    def run(self):
        """Main execution loop for the chat interface.
        
        This method handles:
        1. Page setup and configuration
        2. Display of existing messages with citations
        3. Processing of new user queries
        4. Response generation and display
        5. Citation formatting and presentation
        """
        # Set up the page with all UI elements
        self.setup_page()
        settings = self.setup_sidebar()

        # Display existing messages with their citations
        for message in st.session_state.messages:
            self.display_message(message, settings)
        
        # Handle new user input
        if prompt := st.chat_input("Ask me anything..."):
            # Add user message to the chat
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            # Process query and generate response
            with st.spinner("Thinking..."):
                results = self.process_query(prompt, settings)
                
                # Handle response display
                with st.chat_message("assistant"):
                    if 'error' in results:
                        # Handle error case
                        st.error(f"Error: {results['error']}")
                        response_text = f"I apologize, but I encountered an error: {results['error']}"
                        sources = None
                        
                    else:
                        # Format successful response with citations
                        response_text = results['response']['response_text']
                        sources = self.format_sources(results.get('results', []))
                        
                        # Display response text
                        st.markdown(response_text)
                        
                     # Preprocess the sources string to avoid issues with backslashes
                    if sources and settings["show_sources"]:
                        # Replace newline characters and format the text safely
                        formatted_sources = sources.replace('[', '<b>[').replace(']', ']</b>').replace('\n', '<br>')
                        logger.error (f"  citations  --  {formatted_sources} "  )
                        # Render the sources in the expandable section
                        with st.expander("📚 View Sources", expanded=False):
                            st.markdown(
                                f"""
                                <div class="citation-container">
                                    {formatted_sources}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )


                
                # Add response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": sources,
                    "timestamp": datetime.now().strftime("%H:%M"),
                    "metadata": {
                        "processing_time": f"{results.get('processing_time', 0):.2f}s",
                        "sources_used": len(results.get('results', []))
                    }
                })
                
                # Show processing time
                if 'processing_time' in results:
                    st.caption(f"Response generated in {results['processing_time']:.2f} seconds")

def main():
    """Initialize and run the chat interface."""
    chat_interface = RAGChatInterface()
    chat_interface.run()

if __name__ == "__main__":
    main()
    
    
    
    