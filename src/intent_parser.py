"""
Parse user queries to extract intent and entities.
Uses Claude for natural language understanding.
"""
import json
from typing import Dict, List, Optional

from anthropic import Anthropic

from src.config import ANTHROPIC_API_KEY, MODEL

client = Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are a data discovery assistant for DEECA.
Your job is to understand what environmental data users are looking for.

Extract from user queries:
1. Intent (find_datasets, explore_data, get_specific_dataset)
2. Data type (fire, water, biodiversity, energy, property, climate, etc)
3. Region (default to ["Victoria"] if not specified — do not ask for a region)
4. Constraints (access level, format, recency, etc — leave null if not mentioned)

If earlier turns appear in this conversation, your own prior replies are the
raw JSON you previously returned. If the current message is a follow-up or
refinement (e.g. a narrower topic, a date range, a region, a format) rather
than a full new request, merge it with the data_type/region/constraints from
your most recent prior JSON reply instead of dropping them.

CRITICAL: Only ask for clarification if the query is GENUINELY AMBIGUOUS — i.e.
you cannot infer any data type at all. A short query naming a topic is clear,
not ambiguous. Users want results, not follow-up questions. Make confident
inferences and default missing fields rather than asking.

CLEAR queries (clarification_needed=false) — a topic word or two is enough:
- "fire data" -> data_type: ["fire"], region: ["Victoria"]
- "i want water" -> data_type: ["water"], region: ["Victoria"]
- "biodiversity dandenongs" -> data_type: ["biodiversity"], region: ["Dandenongs"]
- "fire data in victoria from 2010" -> data_type: ["fire"], region: ["Victoria"], constraints.update_recency_days/date info captured

AMBIGUOUS queries (clarification_needed=true) — only when no topic at all is named:
- "show me data" -> no data type mentioned
- "i need information" -> no data type mentioned
- "stuff" -> meaningless

Return ONLY a JSON object (no other text):
{
  "intent": "find_datasets",
  "data_type": ["fire"],
  "region": ["Victoria"],
  "constraints": {
    "access_level": null,
    "format": null,
    "update_recency_days": null
  },
  "clarification_needed": false,
  "clarification_question": null,
  "confidence": 0.85
}"""


FALLBACK_INTENT = {
    "intent": "find_datasets",
    "data_type": [],
    "region": [],
    "constraints": {},
    "clarification_needed": True,
    "clarification_question": "Could you be more specific about what data you're looking for?",
    "confidence": 0.0
}

# Bound how many prior turns are replayed. Each turn is 2 messages (user +
# assistant), so this caps token growth — unbounded history previously
# starved the model's output budget (thinking consumed the whole max_tokens
# before any text was emitted), causing an uncaught crash.
MAX_HISTORY_TURNS = 3


class IntentParser:
    def __init__(self):
        self.model = MODEL

    def parse_intent(self, user_query: str, intent_history: Optional[List[Dict]] = None) -> Dict:
        """
        Parse a user query to extract intent, data_type, region, and
        constraints — using real prior turns as conversation history, the
        way the Anthropic API is designed to be used (alternating user/
        assistant messages), rather than a paraphrased text summary.

        Args:
            user_query: The current message to interpret
            intent_history: Prior turns as alternating {"role", "content"}
                dicts, where each assistant "content" is the raw JSON this
                method previously returned (serialized). Bounded to the
                last MAX_HISTORY_TURNS turns here as a safety net even if
                the caller passes more.

        Returns:
            Dictionary with structured intent information
        """
        messages = list(intent_history or [])[-(MAX_HISTORY_TURNS * 2):]
        messages.append({"role": "user", "content": user_query})

        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages
        )

        text_blocks = [block.text for block in response.content if block.type == "text"]
        if not text_blocks:
            return FALLBACK_INTENT
        assistant_message = "".join(text_blocks)

        try:
            json_start = assistant_message.find('{')
            json_end = assistant_message.rfind('}') + 1
            json_str = assistant_message[json_start:json_end]
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return FALLBACK_INTENT
