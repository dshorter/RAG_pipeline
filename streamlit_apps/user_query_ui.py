import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
import sys, os  

# Project imports    
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.singleton_config import ConfigSingleton
from src.embedding_generator_factory import EmbeddingGeneratorFactory
from src.query_analyzer import QueryAnalyzer
from src.reranker_factory import ReRankerFactory
from src.generation import Generator
from src.rag_search_client import RAGSearchClient
from src.logging_config import get_logger

logger = get_logger('chat')

class RAGChatInterface:
    def __init__(self):
        # Initialize session state for chat
        if 'messages' not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! How can I help you today?"}
            ]
        
        # Initialize RAG components if not already done
        if 'interface' not in st.session_state:
            st.session_state.interface = self._initialize_components()

    def _initialize_components(self) -> Dict:
        """Initialize all RAG components."""
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

    def setup_sidebar(self) -> Dict[str, Any]:
        """Configure chat settings in sidebar."""
        with st.sidebar:
            st.title("Chat Settings")
            
            # Core settings
            settings = {
                "context_size": st.slider("Context Size", 2, 10, 5),
                "use_training_data": st.toggle("Allow Training Data", False, key="toggle_training_data"),
                "show_sources": st.toggle("Show Sources", True, key="toggle_sources"),
                "show_analysis": st.toggle("Show Query Analysis", False, key="toggle_analysis")
            }
            
            # Chat controls
            st.divider()
            if st.button("Clear Chat History"):
                st.session_state.messages = [
                    {"role": "assistant", "content": "Chat history cleared. How can I help?"}
                ]
                st.rerun()
            
            # Display chat statistics
            st.divider()
            st.caption("Chat Statistics")
            user_messages = sum(1 for msg in st.session_state.messages if msg["role"] == "user")
            st.text(f"Messages: {len(st.session_state.messages)}")
            st.text(f"Questions Asked: {user_messages}")
            
            return settings

    def process_query(self, query: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Process user query through RAG pipeline."""
        try:
            start_time = time.time()
            interface = st.session_state.interface
            
            # Analyze query
            analysis = interface['query_analyzer'].analyze_query(query)
            query_vector = interface['embedding_generator'].generate_embedding(query)
            
            # Get search results
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
                final_results = self._format_reranked_results(reranked_results, initial_results)
            else:
                final_results = initial_results[:settings['context_size']]
            
            # Generate response
            response = interface['generator'].generate_response(
                query=query,
                search_results=final_results,
                allow_training_data=True #settings['use_training_data']
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

    def _format_reranked_results(self, reranked_results: List, initial_results: List) -> List[Dict]:
        """Format reranked results with source information."""
        return [{
            'chunk_id': r.chunk_id,
            'chunk_text': r.chunk_text,
            'relevance_score': r.reranked_score,
            'initial_score': r.initial_score,
            'metadata': r.metadata,
            'source_info': next((res['source_info'] for res in initial_results 
                               if res['chunk_id'] == r.chunk_id), {})
        } for r in reranked_results]

    def display_sources(self, results: List[Dict[str, Any]], message_idx: Optional[int] = None):
        """Display source information in an expandable container."""
        with st.expander("View Sources", expanded=False):
            for i, result in enumerate(results, 1):
                source_info = result.get('source_info', {})
                
                # Create columns for source details
                col1, col2 = st.columns([3, 1])
                with col1:
                    title = source_info.get('title', 'Unknown Document')
                    source = source_info.get('source', 'Unknown Source')
                    st.markdown(f"<span style='font-size: 1rem'><b>[{i}] {title}</b> from {source}</span>", 
                              unsafe_allow_html=True)
                
                with col2:
                    score = result.get('relevance_score', 0)
                    st.metric("Relevance", f"{score:.2f}")
                
                # Show snippet with unique key
                toggle_key = f"snippet_{message_idx}_{i}" if message_idx is not None else f"snippet_{i}"
                if st.toggle("Show snippet", False, key=toggle_key):
                    st.markdown(f"<span style='font-size: 0.9rem'>{result['chunk_text'][:200]}...</span>",
                              unsafe_allow_html=True)
                
                st.divider()

    def run(self):
        """Main chat interface loop."""
        # Set up the page layout first
        self.setup_page()
            
        # Get settings from sidebar
        settings = self.setup_sidebar()


        # Display chat messages
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(f"<span style='font-size: 1rem'>{message['content']}</span>", 
                          unsafe_allow_html=True)
                
                # Show sources if available and enabled
                if settings["show_sources"] and "sources" in message:
                    self.display_sources(message.get("sources", []), message_idx=idx)
                
                # Show analysis if enabled
                if settings["show_analysis"] and "analysis" in message:
                    with st.expander("Query Analysis", expanded=False):
                        analysis = message["analysis"]
                        st.info(f"Complexity Score: {analysis.complexity_score:.2f}")
                        st.markdown("<span style='font-size: 0.9rem'>Features:</span>", 
                                  unsafe_allow_html=True)
                        st.json(analysis.features)  # Using st.json for better formatting
        
        # Chat input
        if prompt := st.chat_input("Ask me anything..."):
            # Add user message
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Process query
            with st.spinner("Thinking..."):
                results = self.process_query(prompt, settings)
                
                # Handle response
                with st.chat_message("assistant"):
                    if 'error' in results:
                        st.error(f"Error: {results['error']}")
                        response = f"I apologize, but I encountered an error: {results['error']}"
                        sources = None
                    else:
                        response = results['response']['response_text']
                        st.markdown(f"<span style='font-size: 1rem'>{response}</span>", 
                                  unsafe_allow_html=True)
                        sources = results.get('results', [])
                        
                        if settings["show_sources"]:
                            self.display_sources(sources, message_idx=len(st.session_state.messages))
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources,
                    "analysis": results.get('analysis')
                })
                
                # Show processing time
                if 'processing_time' in results:
                    st.caption(f"Response generated in {results['processing_time']:.2f} seconds")

    def setup_page(self):
        """Configure the Streamlit page layout."""
        st.set_page_config(
            page_title="ChatDRSC - Research Query Interface",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Custom CSS for the interface
        st.markdown("""
            <style>
            /* Core layout */
            .stApp {
                background-color: #ffffff;
                min-height: 100vh;
                padding-bottom: 60px;
            }
            .main > div {
                padding-top: 0 !important;
                margin-top: -2rem;  /* Remove gap at top */
            }
            .block-container {
                padding-top: 0;
                max-width: none;  /* Allow header to go full width */
            }
            
            /* Hide default elements */
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            
            /* Warning notice styling - more subtle with better contrast */
            .stAlert {
                background-color: rgba(254, 243, 199, 0.3) !important;
                border-color: #f59e0b !important;
            }
            .stAlert > div {
                color: #92400e !important;  /* Darker amber for better contrast */
            }
            
            /* Footer styling - adjusted opacity */
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
            
            /* Status indicator */
            .status-dot {
                display: inline-block;
                width: 8px;
                height: 8px;
                background-color: #22c55e;
                border-radius: 50%;
                animation: pulse 2s infinite;
                margin-right: 8px;
            }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            </style>
        """, unsafe_allow_html=True)
        
        #===========================
         # Add our new chat text styling RIGHT HERE
        st.markdown("""
            <style>
                /* Force text color and size for all chat messages */
                .stChatMessage {
                    color: #1e293b !important;
                    font-size: 18px !important;
                }
                .stChatMessage div[data-testid="stMarkdownContainer"] {
                    color: #1e293b !important;
                    font-size: 18px !important;
                }
                .stChatMessage div[data-testid="stMarkdownContainer"] p {
                    color: #1e293b !important;
                    font-size: 18px !important;
                    line-height: 1.6 !important;
                }

                /* Target h2 elements specifically for summaries - ADD THIS HERE */
                .stChatMessage div[data-testid="stMarkdownContainer"] h2,
                .stChatMessage h2 {
                    font-size: 18px !important;
                    line-height: 1.6 !important;
                    font-weight: normal !important;
                    margin-top: 1rem !important;
                    margin-bottom: 1rem !important;
                }

                /* Make sure no other styles override this */
                [data-testid="stMarkdownContainer"] h2 {
                    font-size: 18px !important;
                    font-weight: normal !important;
                }

        /* Source citations slightly smaller */
        .stChatMessage .sources,
        .stChatMessage small {
            font-size: 15px !important;
            color: #64748b !important;
        }
    </style>
""", unsafe_allow_html=True)
        ###########################
        st.markdown("""
            <style>
            /* Force text color for chat messages */
            .stChatMessage {
                color: #1e293b !important;
            }
            .stChatMessage div[data-testid="stMarkdownContainer"] {
                color: #1e293b !important;
            }
            .stChatMessage div[data-testid="stMarkdownContainer"] p {
                color: #1e293b !important;
            }
            
            /* Ensure any user input text is also visible */
            .stTextInput textarea {
                color: #1e293b !important;
            }
            
            /* Force text color for all message containers */
            [data-testid="StyledTheme"] {
                color: #1e293b !important;
            }
            </style>
        """, unsafe_allow_html=True)
        ############################
        # Header - adjusted for full width
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
        
        # Beta Notice - with improved contrast
        st.markdown("""
            <div class="stAlert" style="background-color: rgba(254, 243, 199, 0.3); border-color: #f59e0b; padding: 1rem; border-radius: 0.375rem;">
                <div style="color: #92400e;">
                    🔬 Research Preview: This experimental query interface provides AI-assisted access to research knowledge bases. 
                    Results require validation against primary sources.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Footer - with improved contrast
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
        
        # Sidebar settings
        return {
            "show_analysis": st.sidebar.checkbox("Show Query Analysis", value=True),
            "show_reranking": st.sidebar.checkbox("Show Re-ranking Impact", value=True),
            "context_only": st.sidebar.checkbox("Context Only Mode", value=False)
        }
##############################

def main():
    chat_interface = RAGChatInterface()
    chat_interface.run()

if __name__ == "__main__":
    main()





