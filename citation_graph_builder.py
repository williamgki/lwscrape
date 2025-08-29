#!/usr/bin/env python3

import asyncio
import json
import logging
import aiohttp
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
import time
from datetime import datetime
import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PaperNode:
    """Paper node in citation graph"""
    openalex_id: str
    title: str
    year: Optional[int] = None
    venue: Optional[str] = None
    organization: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    cited_by_count: int = 0
    references_count: int = 0
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    
    # Graph properties
    hop_distance: int = 0  # Distance from target paper
    node_type: str = "reference"  # "target", "reference", "citation"


@dataclass
class CitationEdge:
    """Citation edge in graph"""
    source_id: str
    target_id: str
    edge_type: str  # "cites", "cited_by"
    year_gap: Optional[int] = None


class OpenAlexCitationExpander:
    """Expand citation graph using OpenAlex API"""
    
    def __init__(self, 
                 base_url: str = "https://api.openalex.org",
                 rate_limit_delay: float = 0.1,
                 max_retries: int = 3):
        
        self.base_url = base_url
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        
        # Session for persistent connections
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Request stats
        self.stats = {
            'requests_made': 0,
            'cache_hits': 0,
            'api_errors': 0,
            'papers_fetched': 0
        }
        
        # Simple in-memory cache
        self.paper_cache: Dict[str, PaperNode] = {}
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "AcademicCorpusBuilder/1.0"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_paper_details(self, 
                                 openalex_id: str,
                                 include_abstract: bool = False) -> Optional[PaperNode]:
        """Fetch paper details from OpenAlex"""
        
        # Check cache first
        if openalex_id in self.paper_cache:
            self.stats['cache_hits'] += 1
            return self.paper_cache[openalex_id]
        
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        # Clean OpenAlex ID format
        if not openalex_id.startswith('https://openalex.org/'):
            if openalex_id.startswith('W'):
                openalex_id = f"https://openalex.org/{openalex_id}"
            else:
                openalex_id = f"https://openalex.org/W{openalex_id}"
        
        url = f"{self.base_url}/works/{openalex_id}"
        
        for attempt in range(self.max_retries):
            try:
                await asyncio.sleep(self.rate_limit_delay)  # Rate limiting
                
                async with self.session.get(url) as response:
                    self.stats['requests_made'] += 1
                    
                    if response.status == 200:
                        data = await response.json()
                        paper_node = self._parse_paper_data(data)
                        
                        # Cache the result
                        self.paper_cache[openalex_id] = paper_node
                        self.stats['papers_fetched'] += 1
                        
                        return paper_node
                    
                    elif response.status == 429:  # Rate limited
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    else:
                        logger.warning(f"API error {response.status} for {openalex_id}")
                        self.stats['api_errors'] += 1
                        return None
            
            except Exception as e:
                logger.error(f"Request failed for {openalex_id}: {e}")
                if attempt == self.max_retries - 1:
                    self.stats['api_errors'] += 1
                    return None
        
        return None
    
    def _parse_paper_data(self, data: Dict[str, Any]) -> PaperNode:
        """Parse OpenAlex paper data into PaperNode"""
        
        # Extract basic info
        openalex_id = data.get('id', '')
        title = data.get('title', '')
        year = data.get('publication_year')
        
        # Extract venue information
        venue = None
        host_venue = data.get('primary_location', {})
        if host_venue:
            if host_venue.get('source'):
                venue = host_venue['source'].get('display_name', '')
        
        # Extract authors
        authors = []
        for author in data.get('authorships', []):
            author_info = author.get('author', {})
            display_name = author_info.get('display_name', '')
            if display_name:
                authors.append(display_name)
        
        # Extract organization (first author's affiliation)
        organization = None
        if data.get('authorships'):
            first_author = data['authorships'][0]
            institutions = first_author.get('institutions', [])
            if institutions:
                organization = institutions[0].get('display_name', '')
        
        # Extract DOI and arXiv ID
        doi = None
        arxiv_id = None
        
        for identifier in data.get('ids', {}).items():
            if identifier[0] == 'doi' and identifier[1]:
                doi = identifier[1].replace('https://doi.org/', '')
            elif identifier[0] == 'arxiv' and identifier[1]:
                arxiv_id = identifier[1]
        
        # Citation counts
        cited_by_count = data.get('cited_by_count', 0)
        references_count = len(data.get('referenced_works', []))
        
        return PaperNode(
            openalex_id=openalex_id,
            title=title,
            year=year,
            venue=venue,
            organization=organization,
            authors=authors,
            cited_by_count=cited_by_count,
            references_count=references_count,
            doi=doi,
            arxiv_id=arxiv_id
        )
    
    async def get_references(self, openalex_id: str) -> List[str]:
        """Get reference IDs for a paper"""
        
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        url = f"{self.base_url}/works/{openalex_id}"
        
        try:
            await asyncio.sleep(self.rate_limit_delay)
            
            async with self.session.get(url) as response:
                self.stats['requests_made'] += 1
                
                if response.status == 200:
                    data = await response.json()
                    return data.get('referenced_works', [])
                else:
                    logger.warning(f"Failed to get references for {openalex_id}")
                    return []
        
        except Exception as e:
            logger.error(f"Error getting references for {openalex_id}: {e}")
            return []
    
    async def get_citations(self, 
                          openalex_id: str, 
                          limit: int = 200) -> List[str]:
        """Get citation IDs for a paper"""
        
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        # Use cited_by_api_url from paper details
        paper_details = await self.fetch_paper_details(openalex_id)
        if not paper_details:
            return []
        
        # Build citations query
        url = f"{self.base_url}/works"
        params = {
            'filter': f'cites:{openalex_id}',
            'per-page': min(limit, 200),  # API limit
            'select': 'id'
        }
        
        try:
            await asyncio.sleep(self.rate_limit_delay)
            
            async with self.session.get(url, params=params) as response:
                self.stats['requests_made'] += 1
                
                if response.status == 200:
                    data = await response.json()
                    citations = []
                    for result in data.get('results', []):
                        citations.append(result.get('id', ''))
                    return citations
                else:
                    logger.warning(f"Failed to get citations for {openalex_id}")
                    return []
        
        except Exception as e:
            logger.error(f"Error getting citations for {openalex_id}: {e}")
            return []


class CitationGraphBuilder:
    """Build citation ego-graph around target papers"""
    
    def __init__(self, 
                 storage_path: str = "./citation_graphs",
                 max_papers_per_hop: int = 100):
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.max_papers_per_hop = max_papers_per_hop
        
        # Graph storage
        self.graphs: Dict[str, nx.DiGraph] = {}
        
        # Processing stats
        self.stats = {
            'graphs_built': 0,
            'total_nodes': 0,
            'total_edges': 0,
            'processing_time': 0.0
        }
    
    async def build_ego_graph(self,
                            target_paper_id: str,
                            hops: int = 2,
                            include_citations: bool = True,
                            include_references: bool = True,
                            min_survey_refs: int = 5) -> nx.DiGraph:
        """Build citation ego-graph around target paper"""
        
        logger.info(f"Building {hops}-hop ego-graph for {target_paper_id}")
        start_time = time.time()
        
        # Initialize graph
        graph = nx.DiGraph()
        
        async with OpenAlexCitationExpander() as expander:
            
            # Add target paper as root node
            target_node = await expander.fetch_paper_details(target_paper_id)
            if not target_node:
                logger.error(f"Could not fetch target paper: {target_paper_id}")
                return graph
            
            target_node.node_type = "target"
            target_node.hop_distance = 0
            
            graph.add_node(target_paper_id, **target_node.__dict__)
            
            # Expand outward for specified number of hops
            current_level = {target_paper_id}
            
            for hop in range(1, hops + 1):
                next_level = set()
                
                logger.info(f"Processing hop {hop} with {len(current_level)} papers")
                
                # Process each paper in current level
                for paper_id in current_level:
                    
                    # Get references (papers this paper cites)
                    if include_references:
                        refs = await expander.get_references(paper_id)
                        refs = refs[:self.max_papers_per_hop]  # Limit for performance
                        
                        for ref_id in refs:
                            if ref_id and ref_id not in graph:
                                # Fetch reference details
                                ref_node = await expander.fetch_paper_details(ref_id)
                                if ref_node:
                                    ref_node.node_type = "reference"
                                    ref_node.hop_distance = hop
                                    graph.add_node(ref_id, **ref_node.__dict__)
                                    next_level.add(ref_id)
                            
                            # Add citation edge
                            if ref_id and ref_id in graph:
                                year_gap = self._calculate_year_gap(
                                    graph.nodes[paper_id].get('year'),
                                    graph.nodes[ref_id].get('year')
                                )
                                graph.add_edge(paper_id, ref_id, 
                                             edge_type="cites", year_gap=year_gap)
                    
                    # Get citations (papers that cite this paper)
                    if include_citations:
                        cites = await expander.get_citations(paper_id, limit=self.max_papers_per_hop)
                        
                        for cite_id in cites:
                            if cite_id and cite_id not in graph:
                                # Fetch citation details
                                cite_node = await expander.fetch_paper_details(cite_id)
                                if cite_node:
                                    cite_node.node_type = "citation"
                                    cite_node.hop_distance = hop
                                    graph.add_node(cite_id, **cite_node.__dict__)
                                    next_level.add(cite_id)
                            
                            # Add citation edge
                            if cite_id and cite_id in graph:
                                year_gap = self._calculate_year_gap(
                                    graph.nodes[cite_id].get('year'),
                                    graph.nodes[paper_id].get('year')
                                )
                                graph.add_edge(cite_id, paper_id,
                                             edge_type="cites", year_gap=year_gap)
                
                current_level = next_level
                
                if len(current_level) == 0:
                    logger.info(f"No new papers found at hop {hop}, stopping")
                    break
            
            # Store final API stats
            logger.info(f"OpenAlex API stats: {expander.stats}")
        
        # Mark survey/tutorial nodes with too few references
        low_info_nodes = self._check_survey_outgoing_refs(graph, min_survey_refs)
        if low_info_nodes:
            logger.warning(f"Found {len(low_info_nodes)} low-info survey/tutorial nodes")

        # Store graph
        self.graphs[target_paper_id] = graph
        
        # Update stats
        self.stats['graphs_built'] += 1
        self.stats['total_nodes'] += graph.number_of_nodes()
        self.stats['total_edges'] += graph.number_of_edges()
        self.stats['processing_time'] += time.time() - start_time
        
        logger.info(f"Built ego-graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        return graph

    def _check_survey_outgoing_refs(self, graph: nx.DiGraph, min_refs: int) -> List[str]:
        """Mark survey/tutorial nodes with insufficient outgoing references"""

        low_info = []
        for node_id, data in graph.nodes(data=True):
            title = data.get('title', '').lower()
            if 'survey' in title or 'tutorial' in title:
                out_refs = graph.out_degree(node_id)
                if out_refs < min_refs:
                    graph.nodes[node_id]['low_info'] = True
                    low_info.append(node_id)
                else:
                    graph.nodes[node_id]['low_info'] = False
        return low_info
    
    def _calculate_year_gap(self, cite_year: Optional[int], ref_year: Optional[int]) -> Optional[int]:
        """Calculate year gap between citing and referenced paper"""
        if cite_year and ref_year:
            return cite_year - ref_year
        return None
    
    def analyze_graph(self, target_paper_id: str) -> Dict[str, Any]:
        """Analyze citation graph properties"""
        
        if target_paper_id not in self.graphs:
            return {"error": "Graph not found"}
        
        graph = self.graphs[target_paper_id]
        
        analysis = {
            'target_paper_id': target_paper_id,
            'total_nodes': graph.number_of_nodes(),
            'total_edges': graph.number_of_edges(),
            'node_types': {},
            'hop_distribution': {},
            'year_distribution': {},
            'venue_distribution': {},
            'organization_distribution': {},
            'centrality_measures': {}
        }
        
        # Analyze node properties
        for node_id, node_data in graph.nodes(data=True):
            # Node types
            node_type = node_data.get('node_type', 'unknown')
            analysis['node_types'][node_type] = analysis['node_types'].get(node_type, 0) + 1
            
            # Hop distribution
            hop = node_data.get('hop_distance', 0)
            analysis['hop_distribution'][f"hop_{hop}"] = analysis['hop_distribution'].get(f"hop_{hop}", 0) + 1
            
            # Year distribution
            year = node_data.get('year')
            if year:
                decade = f"{year//10*10}s"
                analysis['year_distribution'][decade] = analysis['year_distribution'].get(decade, 0) + 1
            
            # Venue distribution
            venue = node_data.get('venue')
            if venue:
                analysis['venue_distribution'][venue] = analysis['venue_distribution'].get(venue, 0) + 1
            
            # Organization distribution
            org = node_data.get('organization')
            if org:
                analysis['organization_distribution'][org] = analysis['organization_distribution'].get(org, 0) + 1
        
        # Calculate centrality measures
        try:
            analysis['centrality_measures'] = {
                'in_degree_centrality': nx.in_degree_centrality(graph),
                'out_degree_centrality': nx.out_degree_centrality(graph),
                'betweenness_centrality': nx.betweenness_centrality(graph)
            }
        except Exception as e:
            logger.warning(f"Failed to calculate centrality measures: {e}")
        
        return analysis
    
    def save_graph(self, target_paper_id: str):
        """Save citation graph to files"""
        
        if target_paper_id not in self.graphs:
            logger.error(f"Graph not found: {target_paper_id}")
            return
        
        graph = self.graphs[target_paper_id]
        
        # Clean paper ID for filename
        clean_id = target_paper_id.replace('https://openalex.org/', '').replace('/', '_')
        
        # Save as JSON
        json_file = self.storage_path / f"{clean_id}_graph.json"
        graph_data = {
            'target_paper_id': target_paper_id,
            'created_timestamp': datetime.now().isoformat(),
            'nodes': {
                node_id: node_data 
                for node_id, node_data in graph.nodes(data=True)
            },
            'edges': [
                {
                    'source': source,
                    'target': target,
                    'edge_type': edge_data.get('edge_type', 'cites'),
                    'year_gap': edge_data.get('year_gap')
                }
                for source, target, edge_data in graph.edges(data=True)
            ]
        }
        
        with open(json_file, 'w') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        # Save analysis
        analysis_file = self.storage_path / f"{clean_id}_analysis.json"
        analysis = self.analyze_graph(target_paper_id)
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved citation graph: {json_file}")
        logger.info(f"Saved analysis: {analysis_file}")
    
    def get_paper_context(self, 
                         target_paper_id: str, 
                         context_type: str = "influential") -> List[str]:
        """Get contextual papers from the ego-graph"""
        
        if target_paper_id not in self.graphs:
            return []
        
        graph = self.graphs[target_paper_id]
        
        if context_type == "influential":
            # Papers with high citation counts
            papers = [
                (node_id, node_data.get('cited_by_count', 0))
                for node_id, node_data in graph.nodes(data=True)
                if node_id != target_paper_id
            ]
            papers.sort(key=lambda x: x[1], reverse=True)
            return [paper[0] for paper in papers[:20]]
        
        elif context_type == "recent":
            # Recent papers (within 3 years)
            target_year = graph.nodes[target_paper_id].get('year', 2020)
            recent_papers = [
                (node_id, node_data.get('year', 0))
                for node_id, node_data in graph.nodes(data=True)
                if node_data.get('year', 0) >= target_year - 3
            ]
            recent_papers.sort(key=lambda x: x[1], reverse=True)
            return [paper[0] for paper in recent_papers[:20]]
        
        elif context_type == "foundational":
            # Highly referenced papers from earlier years
            target_year = graph.nodes[target_paper_id].get('year', 2020)
            foundational = [
                (node_id, node_data.get('cited_by_count', 0))
                for node_id, node_data in graph.nodes(data=True)
                if node_data.get('year', 2030) < target_year - 2
            ]
            foundational.sort(key=lambda x: x[1], reverse=True)
            return [paper[0] for paper in foundational[:20]]
        
        return []


async def main():
    """Demo citation graph building"""
    
    builder = CitationGraphBuilder()
    
    print("=== Citation Graph Builder Demo ===")
    print("\nC4. Build citation graph slice:")
    print("✓ Use DocRefEdges + OpenAlex API")
    print("✓ Expand 1-2 hops around target paper")
    print("✓ Persist ego-graph with node attributes")
    print("✓ Include year, venue, organization metadata")
    
    # Example target paper (would use real OpenAlex ID)
    target_paper = "https://openalex.org/W2741809807"  # Attention is All You Need
    
    print(f"\nBuilding ego-graph for: {target_paper}")
    print("This would:")
    print("1. Fetch target paper details from OpenAlex")
    print("2. Get all referenced papers (1st hop backward)")
    print("3. Get all citing papers (1st hop forward)")
    print("4. Optionally expand to 2nd hop")
    print("5. Add node attributes: year, venue, organization")
    print("6. Calculate centrality measures")
    print("7. Save as JSON with analysis")
    
    example_node = {
        "openalex_id": "https://openalex.org/W2741809807",
        "title": "Attention Is All You Need",
        "year": 2017,
        "venue": "Neural Information Processing Systems",
        "organization": "Google",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "cited_by_count": 45000,
        "node_type": "target",
        "hop_distance": 0
    }
    
    print(f"\nExample node structure:")
    print(json.dumps(example_node, indent=2))


if __name__ == "__main__":
    asyncio.run(main())