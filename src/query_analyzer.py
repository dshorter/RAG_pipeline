
import spacy
from typing import List, Dict
import re
from datetime import datetime
from src.singleton_config import ConfigSingleton
from src.logging_config import get_logger
from src.data_classes import QueryAnalysisResult  
from src.metrics_collector  import  MetricsCollector  
import time

class QueryAnalyzer:
    def __init__(self):
        self.config = ConfigSingleton()
        self.logger = get_logger('analyzer')  
        
        self.metrics_collector = MetricsCollector( )          
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("Downloading spacy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        analysis_config = self.config.get_query_analysis_config()
        self.thresholds = analysis_config.thresholds
        self.weights = analysis_config.weights
        
        self.complexity_indicators = {
            'compare': 1.5, 'difference': 1.5, 'relationship': 1.5,
            'versus': 1.5, 'impact': 1.2, 'explain': 1.2,
            'why': 1.2, 'how': 1.2, 'analyze': 1.5, 'evaluate': 1.5
        }
        
        self.simplicity_indicators = {
            'what is': 0.8, 'where is': 0.8, 'when': 0.8,
            'who': 0.8, 'list': 0.8, 'find': 0.8
        }

    def analyze_query(self, query: str) -> QueryAnalysisResult:
        start_time = time.time()
        try:
            doc = self.nlp(query.lower())
            
            features = {
                'word_count': len(doc) * self.weights['word_count'],
                'entity_count': len(doc.ents) * self.weights['entity_count'],
                'dependency_depth': self._calculate_dependency_depth(doc) * self.weights['dependency_depth'],
                'keyword_complexity': self._calculate_keyword_complexity(query) * self.weights['keyword_complexity'],
                'question_complexity': self._analyze_question_type(query) * self.weights['question_complexity']
            }
            
            complexity_score = sum(features.values())
            needs_reranking = complexity_score > self.thresholds['complex_query']
            
            if complexity_score > self.thresholds['complex_query']:
                recommended_k = self.thresholds['max_chunks_complex']
                recommended_candidates = self.thresholds['max_candidates']
            else:
                recommended_k = self.thresholds['max_chunks_simple']
                recommended_candidates = recommended_k * 2

            explanation = self._generate_explanation(features, complexity_score)

            self.metrics_collector.collect(
                operation='query_analysis',
                component='query_analyzer',
                metrics={
                    'duration_ms': (time.time() - start_time) * 1000,
                    'complexity_score': complexity_score,
                    'needs_reranking': needs_reranking,
                    'word_count': len(doc),
                    'entity_count': len(doc.ents),
                    'features': features,
                    'model': {
                        'name': 'spacy-sm',
                        'entities_found': [str(ent.label_) for ent in doc.ents]
                    }
                }
            )
            
            return QueryAnalysisResult(
                complexity_score=complexity_score,
                needs_reranking=needs_reranking,
                recommended_k=recommended_k,
                recommended_candidates=recommended_candidates,
                features=features,
                explanation=explanation
            )
            
        except Exception as e:
            self.logger.error(f"Query analysis failed: {str(e)}")
            self.metrics_collector.collect(
                operation='query_analysis',
                component='query_analyzer',
                metrics={
                    'duration_ms': (time.time() - start_time) * 1000,
                    'success': False,
                    'error': str(e)
                }
            )
            raise

    def _calculate_dependency_depth(self, doc) -> float:
        max_depth = 0
        for token in doc:
            depth = 1
            current = token
            while current.head != current:
                depth += 1
                current = current.head
            max_depth = max(max_depth, depth)
        return max_depth / 5

    def _calculate_keyword_complexity(self, query: str) -> float:
        score = 1.0
        lower_query = query.lower()
        
        for word, weight in self.complexity_indicators.items():
            if word in lower_query:
                score *= weight
        
        for phrase, weight in self.simplicity_indicators.items():
            if phrase in lower_query:
                score *= weight
        
        return score

    def _analyze_question_type(self, query: str) -> float:
        lower_query = query.lower()
        
        if len(re.findall(r'\?', query)) > 1:
            return 2.0
            
        if any(phrase in lower_query for phrase in [
            'how does', 'in what way', 'what is the relationship',
            'what are the implications', 'to what extent'
        ]):
            return 1.5
            
        if any(phrase in lower_query for phrase in [
            'what is', 'where is', 'when did', 'who is'
        ]):
            return 0.8
            
        return 1.0

    def _generate_explanation(self, features: Dict[str, float], complexity_score: float) -> List[str]:
        explanations = []
        
        if features['word_count'] > 1.0:
            explanations.append("Query is relatively long")
        else:
            explanations.append("Query is concise")
        
        if features['entity_count'] > 0.4:
            explanations.append("Query contains multiple named entities")
        
        if features['dependency_depth'] > 0.6:
            explanations.append("Query has complex sentence structure")
        else:
            explanations.append("Query has simple sentence structure")
        
        if features['keyword_complexity'] > 1.2:
            explanations.append("Query contains terms suggesting complex relationships")
        elif features['keyword_complexity'] < 0.8:
            explanations.append("Query appears to be seeking factual information")
        
        if complexity_score > self.thresholds['complex_query']:
            explanations.append("Re-ranking recommended for better semantic understanding")
        else:
            explanations.append("Simple vector search should be sufficient")
        
        return explanations    
    
    