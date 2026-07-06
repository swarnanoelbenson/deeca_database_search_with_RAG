"""
Search for datasets based on semantic similarity and filters.
"""
from typing import Dict, List

from src.database import Dataset, get_session
from src.embeddings import generate_embedding


class SemanticSearch:
    def search(
        self,
        query: str,
        intent: Dict,
        limit: int = 5
    ) -> List[Dict]:
        """
        Search for datasets using:
        1. Semantic similarity (pgvector cosine distance) as primary retrieval
        2. Category/region/format matches as a re-ranking boost
        3. Access level as a hard filter (a real access constraint, not a guess)

        Real CKAN tags (e.g. "Inland waters") rarely match the intent
        parser's generic data_type guesses (e.g. "water") exactly, so
        category/region matching is done as fuzzy substring scoring rather
        than a hard SQL filter — otherwise a single guessed keyword can
        wipe out every result.

        Args:
            query: User's natural language query
            intent: Parsed intent from IntentParser
            limit: Max number of results to return

        Returns:
            List of matching datasets with metadata, ranked by relevance
        """
        session = get_session()

        query_embedding = generate_embedding(query)
        distance = Dataset.description_embedding.cosine_distance(query_embedding)

        # Rank by vector similarity first (pgvector cosine distance operator)
        base_query = session.query(Dataset, distance.label('distance')).order_by(distance)

        constraints = intent.get('constraints', {})
        if constraints.get('access_level'):
            base_query = base_query.filter(
                Dataset.access_level.in_([constraints['access_level'], 'PUBLIC'])
            )

        # Pull a larger candidate pool than `limit` so keyword-based scoring
        # below can re-rank within the semantically-nearest results.
        candidates = base_query.limit(max(limit * 4, 20)).all()
        session.close()

        data_types = [t.lower() for t in intent.get('data_type', [])]
        regions = [r.lower() for r in intent.get('region', [])]

        scored = []
        for dataset, distance in candidates:
            score = 0.0

            categories = [c.lower() for c in (dataset.category or [])]
            if any(dt in cat or cat in dt for dt in data_types for cat in categories):
                score += 3

            coverage = [c.lower() for c in (dataset.coverage or [])]
            if any(rg in cov or cov in rg for rg in regions for cov in coverage):
                score += 2

            if constraints.get('access_level') and dataset.access_level == constraints['access_level']:
                score += 1

            if constraints.get('format'):
                if any(fmt in (dataset.formats or []) for fmt in constraints['format']):
                    score += 1

            # Blend in semantic similarity so vector rank isn't discarded
            # by keyword scoring alone (closer distance = higher score).
            score += max(0.0, 1.0 - distance)

            scored.append((dataset, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for dataset, relevance_score in scored[:limit]:
            item = dataset.to_dict()
            item['relevance_score'] = round(relevance_score, 3)
            results.append(item)

        return results
