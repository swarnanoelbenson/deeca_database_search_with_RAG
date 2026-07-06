import pytest

from src.intent_parser import IntentParser


@pytest.mark.parametrize("query", [
    "Find water quality data for the Murray",
    "Show me biodiversity datasets",
    "I need fire history for Victoria",
    "What do you have on endangered species in the Dandenongs?",
])
def test_intent_parsing_shape(query):
    parser = IntentParser()
    result = parser.parse_intent(query)
    assert 'intent' in result
    assert 'data_type' in result
    assert 'region' in result
