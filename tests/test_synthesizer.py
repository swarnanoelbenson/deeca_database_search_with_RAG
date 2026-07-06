from src.synthesizer import DatasetSynthesizer

MOCK_DATASETS = [
    {
        "id": "test1",
        "name": "test_dataset",
        "title": "Test Dataset",
        "owner": "Test Owner",
        "description": "A test dataset for biodiversity",
        "access_level": "PUBLIC",
        "formats": ["CSV"],
        "last_updated": "2024-01-01",
        "url": "https://example.com",
        "coverage": ["Victoria"],
        "category": ["biodiversity"]
    }
]


def test_synthesis_produces_text():
    synthesizer = DatasetSynthesizer()

    response = synthesizer.synthesize_response(
        "Find biodiversity data",
        MOCK_DATASETS,
        {}
    )

    assert len(response) > 0


def test_synthesis_empty_results():
    synthesizer = DatasetSynthesizer()
    response = synthesizer.synthesize_response("Find biodiversity data", [], {})
    assert "couldn't find" in response.lower()
