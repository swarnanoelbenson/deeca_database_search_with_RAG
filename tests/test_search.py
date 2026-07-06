import pytest

from src.database import init_db
from src.semantic_search import SemanticSearch


@pytest.fixture(scope="module")
def setup_db():
    init_db()
    yield


def test_semantic_search_returns_list(setup_db):
    search = SemanticSearch()

    intent = {
        "intent": "find_datasets",
        "data_type": ["biodiversity"],
        "region": [],
        "constraints": {}
    }

    results = search.search("biodiversity data", intent)
    assert isinstance(results, list)
