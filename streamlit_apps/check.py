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
            st.session_state.chat_store = ChatHistoryStore()
            
            # Instance methods/properties
            self.config = st.session_state.config
            self.logger = st.session_state.logger
            self.query_analyzer = st.session_state.query_analyzer
            self.reranker = st.session_state.reranker
            self.search_client = st.session_state.search_client
            self.embedding_generator = st.session_state.embedding_generator
            self.generator = st.session_state.generator
            self.chat_store = st.session_state.chat_store

            # Define the process_query method
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

            # Bind the process_query method to the instance
            self.process_query = process_query.__get__(self, RAGQueryUI)
            
            st.session_state.initialized = True
            
            if hasattr(st.session_state, 'logger'):
                st.session_state.logger.info(
                    "Session state initialized", 
                    extra={'component': 'ui', 'operation': 'initialization'}
                )
        except Exception as e:
            st.error(f"Failed to initialize components: {str(e)}")
            raise