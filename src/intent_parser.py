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
2. Data type (biodiversity, water, fire, energy, property, climate, etc)
3. Region/location (Victoria-wide, specific region, suburb, etc)
4. Constraints (access level needed, format preference, recency, etc)

Return ONLY a JSON object (no other text):
{
  "intent": "find_datasets",
  "data_type": ["biodiversity"],
  "region": ["Dandenongs"],
  "constraints": {
    "access_level": "PUBLIC",
    "format": ["CSV"],
    "update_recency_days": null
  },
  "clarification_needed": false,
  "clarification_question": null,
  "confidence": 0.95
}

If you need clarification, set clarification_needed=true."""


FALLBACK_INTENT = {
    "intent": "find_datasets",
    "data_type": [],
    "region": [],
    "constraints": {},
    "clarification_needed": True,
    "clarification_question": "Could you be more specific about what data you're looking for?",
    "confidence": 0.0
}


class IntentParser:
    def __init__(self):
        self.model = MODEL

    def parse_intent(self, user_query: str, recent_user_messages: Optional[List[str]] = None) -> Dict:
        """
        Parse a user query to extract intent, data_type, region, and
        constraints.

        Each call is otherwise independent — no accumulated, ever-growing
        conversation history is sent (that caused unbounded token growth
        and eventually starved the model's output budget). Instead, the
        caller may pass a small, bounded list of recent prior user messages
        (e.g. the last 1-2 turns) so a follow-up like "after 2010" can be
        merged with the data type/region mentioned earlier, without
        replaying the full transcript.

        Args:
            user_query: The current message to interpret
            recent_user_messages: Up to a couple of prior user messages,
                oldest first, for follow-up disambiguation only

        Returns:
            Dictionary with structured intent information
        """
        if recent_user_messages:
            context = "\n".join(f"- \"{m}\"" for m in recent_user_messages)
            composed_query = (
                f"Earlier messages in this conversation (context only):\n{context}\n\n"
                f"Current message to interpret: \"{user_query}\"\n\n"
                "If the current message is a follow-up/refinement (e.g. a date range, "
                "format, or access level with no data type/region of its own), merge in "
                "the data type/region from the earlier messages above."
            )
        else:
            composed_query = user_query

        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": composed_query}]
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
