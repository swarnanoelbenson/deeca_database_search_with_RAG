"""
Manual diagnostic script to verify chat continuity through the real pipeline
(IntentParser with bounded real-turn history, SemanticSearch, DatasetSynthesizer).

Run with: python -m src.diagnose_continuity
"""
import json
import logging

from src.intent_parser import IntentParser
from src.semantic_search import SemanticSearch
from src.synthesizer import DatasetSynthesizer

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_turn(parser, searcher, synthesizer, message, intent_history):
    print(f"\n>>> User: '{message}'")
    print("-" * 80)

    intent = parser.parse_intent(message, intent_history)
    print("Intent parsed:")
    print(f"  data_type: {intent.get('data_type')}")
    print(f"  region: {intent.get('region')}")
    print(f"  constraints: {intent.get('constraints')}")
    print(f"  clarification_needed: {intent.get('clarification_needed')}")

    if intent.get('clarification_needed'):
        print(f"  -> Clarification: {intent.get('clarification_question')}")
        results = []
    else:
        recent_user_messages = [t["content"] for t in intent_history if t.get("role") == "user"]
        search_query = " ".join(recent_user_messages + [message])
        results = searcher.search(search_query, intent, limit=5)
        print(f"Search returned {len(results)} results")
        if results:
            print(f"  Top result: {results[0]['title']} (relevance={results[0]['relevance_score']})")

    intent_history = intent_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": json.dumps(intent)},
    ]
    return intent, results, intent_history


def diagnose_continuity():
    print("\n" + "=" * 80)
    print("CHAT CONTINUITY DIAGNOSTIC TEST")
    print("=" * 80)

    parser = IntentParser()
    searcher = SemanticSearch()
    synthesizer = DatasetSynthesizer()

    intent_history = []

    logger.info("Turn 1: water data")
    intent1, results1, intent_history = run_turn(
        parser, searcher, synthesizer, "i need water data", intent_history
    )

    logger.info("Turn 2: only bedrock - context passed, intent refined")
    intent2, results2, intent_history = run_turn(
        parser, searcher, synthesizer, "only bedrock", intent_history
    )

    print("\n" + "=" * 80)
    print("CONTINUITY CHECK:")
    print("=" * 80)

    data_types_2 = [str(dt).lower() for dt in intent2.get('data_type', [])]
    constraints_2 = intent2.get('constraints', {})

    has_water = any('water' in dt for dt in data_types_2)
    has_bedrock_signal = any(
        'bedrock' in dt for dt in data_types_2
    ) or any(
        'bedrock' in str(v).lower() for v in constraints_2.values() if v
    )

    print(f"\nTurn 2 data_type: {intent2.get('data_type')}")
    print(f"Turn 2 constraints: {constraints_2}")
    print(f"\n water carried forward: {has_water}")
    print(f"bedrock signal present:  {has_bedrock_signal}")

    print("\n" + "-" * 80)
    if has_water and has_bedrock_signal:
        print("CONTINUITY IS WORKING: 'water' + 'only bedrock' combined correctly.")
    elif has_water and not has_bedrock_signal:
        print("PARTIAL: 'water' carried forward, but no bedrock signal found in this run.")
    elif not has_water and has_bedrock_signal:
        print("BROKEN: 'water' was lost, only bedrock remained.")
    else:
        print("BROKEN: neither water nor bedrock present.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    diagnose_continuity()
