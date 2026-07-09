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

        # Default to Victoria-wide if no region was specified. This is a
        # scoring default only, not a filter — every dataset currently has
        # coverage=['Victoria'] anyway (see data_fetcher.py normalize_dataset),
        # so this keeps behavior correct/neutral now and does the right
        # thing automatically once per-region coverage data is added later.
        regions = [r.lower() for r in (intent.get('region') or ['Victoria'])]

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

            # Any other free-form constraint the intent parser extracted
            # (e.g. a refinement like "only bedrock" -> {"subtype": "bedrock
            # aquifer"}) — the schema only has fixed keys for access_level/
            # format/update_recency_days, so anything else is domain-specific
            # and scored generically: does it appear in the description?
            # Without this, a refining constraint is parsed correctly but
            # has zero effect on ranking, so it can't break a near-tie in
            # favor of the dataset that actually matches the refinement.
            handled_keys = {'access_level', 'format', 'update_recency_days'}
            description_lower = (dataset.description or "").lower()
            title_lower = dataset.title.lower()
            for key, value in constraints.items():
                if key in handled_keys or not value or not isinstance(value, str):
                    continue
                # Score each significant word of the constraint separately
                # (e.g. "bedrock aquifer" -> "bedrock", "aquifer") so a
                # dataset matching part of a multi-word refinement still
                # gets credit, rather than requiring an exact phrase match.
                words = [w for w in value.lower().split() if len(w) > 3]
                for word in words:
                    if word in title_lower:
                        score += 2
                    elif word in description_lower:
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
