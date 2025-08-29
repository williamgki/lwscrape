#!/usr/bin/env python3
"""
Corpus Relevance Scorer
Scores chunks by AI relevance for rapid surfacing of most important content
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import json
from typing import Dict, List, Tuple
import logging
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorpusRelevanceScorer:
    def __init__(self):
        # AI relevance keywords with weights
        self.ai_keywords = {
            'critical': 2.0,  # Core AI concepts
            'high': 1.5,      # Important AI topics  
            'medium': 1.0,    # Related concepts
            'low': 0.5        # Tangentially related
        }
        
        self.keyword_categories = {
            'critical': [
                'artificial general intelligence', 'agi', 'superintelligence', 
                'ai alignment', 'ai safety', 'existential risk', 'mesa-optimizer',
                'inner alignment', 'outer alignment', 'reward hacking', 'deception'
            ],
            'high': [
                'artificial intelligence', 'machine learning', 'deep learning',
                'neural network', 'transformer', 'large language model', 'llm',
                'reinforcement learning', 'gradient descent', 'backpropagation'
            ],
            'medium': [
                'algorithm', 'optimization', 'training', 'inference', 'embedding',
                'attention', 'generative', 'supervised learning', 'unsupervised'
            ],
            'low': [
                'automation', 'data science', 'statistics', 'computer vision',
                'natural language processing', 'nlp'
            ]
        }
        
        # Domain authority scores
        self.domain_scores = {
            'arxiv.org': 1.0,
            'openai.com': 0.95,
            'anthropic.com': 0.95, 
            'deepmind.com': 0.95,
            'lesswrong.com': 0.9,
            'alignmentforum.org': 0.9,
            'distill.pub': 0.85,
            'ai.googleblog.com': 0.8,
            'blog.openai.com': 0.8,
            'github.com': 0.7,
            'medium.com': 0.6,
            'reddit.com': 0.4
        }
        
        # Structure type weights
        self.structure_weights = {
            'academic': 1.0,
            'methodology': 0.95,
            'results': 0.9,
            'introduction': 0.8,
            'summary': 0.85,
            'content': 0.7,
            'heading': 0.6,
            'paragraph': 0.5
        }
    
    def score_content_relevance(self, content: str, summary: str = "") -> float:
        """Score content based on AI keyword density"""
        text = f"{content} {summary}".lower()
        
        total_score = 0
        word_count = len(text.split())
        
        for category, weight in self.ai_keywords.items():
            keywords = self.keyword_categories[category]
            matches = sum(len(re.findall(rf'\b{keyword}\b', text)) for keyword in keywords)
            category_score = (matches / max(word_count, 1)) * weight * 1000  # Scale up
            total_score += category_score
        
        return min(total_score, 1.0)  # Cap at 1.0
    
    def score_domain_authority(self, url: str) -> float:
        """Score based on domain authority"""
        if not url:
            return 0.5
            
        domain = urlparse(url).netloc.lower()
        
        # Exact domain match
        if domain in self.domain_scores:
            return self.domain_scores[domain]
        
        # Subdomain or partial matches
        for known_domain, score in self.domain_scores.items():
            if known_domain in domain or domain.endswith(known_domain):
                return score
        
        # Academic institutions (.edu, .ac.uk, etc.)
        if '.edu' in domain or '.ac.' in domain:
            return 0.8
        
        # Government/research (.gov, .org research)
        if '.gov' in domain:
            return 0.75
        
        return 0.5  # Default score
    
    def score_structure_type(self, structure_type: str) -> float:
        """Score based on document structure type"""
        return self.structure_weights.get(structure_type, 0.5)
    
    def score_ai_summary_quality(self, summary_header: str) -> float:
        """Score based on AI summary content quality"""
        if not summary_header:
            return 0.5
            
        # Check for AI-specific terms in summary
        ai_indicators = [
            'artificial intelligence', 'machine learning', 'neural', 'algorithm',
            'training', 'model', 'optimization', 'alignment', 'safety'
        ]
        
        summary_lower = summary_header.lower()
        matches = sum(1 for term in ai_indicators if term in summary_lower)
        
        return min(matches / len(ai_indicators), 1.0)
    
    def calculate_composite_score(self, chunk: pd.Series) -> float:
        """Calculate final composite relevance score"""
        
        # Individual component scores
        content_score = self.score_content_relevance(
            chunk.get('content', ''), 
            chunk.get('summary_header', '')
        )
        
        domain_score = self.score_domain_authority(chunk.get('url', ''))
        
        structure_score = self.score_structure_type(chunk.get('structure_type', ''))
        
        summary_score = self.score_ai_summary_quality(chunk.get('summary_header', ''))
        
        # Weighted composite score
        composite = (
            content_score * 0.4 +      # Content relevance is most important
            domain_score * 0.3 +       # Domain authority is significant  
            summary_score * 0.2 +      # AI summary quality matters
            structure_score * 0.1      # Structure type is least weighted
        )
        
        return round(composite, 4)
    
    def score_corpus(self, corpus_path: str) -> pd.DataFrame:
        """Score entire corpus for AI relevance"""
        logger.info(f"🎯 Loading corpus: {corpus_path}")
        df = pd.read_parquet(corpus_path)
        
        logger.info(f"📊 Scoring {len(df):,} chunks for AI relevance...")
        
        # Calculate relevance scores
        df['ai_relevance_score'] = df.apply(self.calculate_composite_score, axis=1)
        
        # Add percentile ranking
        df['ai_relevance_percentile'] = df['ai_relevance_score'].rank(pct=True)
        
        # Sort by relevance (highest first)
        df_scored = df.sort_values('ai_relevance_score', ascending=False)
        
        # Generate statistics
        stats = {
            'total_chunks': len(df_scored),
            'mean_relevance_score': df_scored['ai_relevance_score'].mean(),
            'top_10_percent_threshold': df_scored['ai_relevance_score'].quantile(0.9),
            'top_domains': df_scored.groupby('domain')['ai_relevance_score'].mean().nlargest(10).to_dict(),
            'top_structure_types': df_scored.groupby('structure_type')['ai_relevance_score'].mean().to_dict()
        }
        
        logger.info("="*60)
        logger.info("🎯 AI RELEVANCE SCORING COMPLETE")
        logger.info("="*60)
        logger.info(f"📊 Mean relevance score: {stats['mean_relevance_score']:.4f}")
        logger.info(f"🔝 Top 10% threshold: {stats['top_10_percent_threshold']:.4f}")
        logger.info(f"🏆 Highest scoring domain: {list(stats['top_domains'].keys())[0]}")
        logger.info("="*60)
        
        return df_scored, stats

def main():
    scorer = CorpusRelevanceScorer()
    
    # Score the final unified corpus
    corpus_path = "/home/ubuntu/LW_scrape/final_combined_corpus/final_unified_corpus.parquet"
    
    if not Path(corpus_path).exists():
        logger.error(f"Corpus not found: {corpus_path}")
        return
    
    # Score corpus
    scored_corpus, stats = scorer.score_corpus(corpus_path)
    
    # Save scored corpus
    output_dir = Path("/home/ubuntu/LW_scrape/scored_corpus")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "ai_relevance_scored_corpus.parquet"
    scored_corpus.to_parquet(output_file, index=False)
    
    # Save statistics
    stats_file = output_dir / "ai_relevance_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    # Save top chunks for quick access
    top_chunks = scored_corpus.head(1000)
    top_file = output_dir / "top_1000_ai_relevant_chunks.parquet"
    top_chunks.to_parquet(top_file, index=False)
    
    logger.info(f"💾 Saved scored corpus: {output_file}")
    logger.info(f"📊 Saved statistics: {stats_file}")  
    logger.info(f"🔝 Saved top 1000 chunks: {top_file}")

if __name__ == "__main__":
    main()