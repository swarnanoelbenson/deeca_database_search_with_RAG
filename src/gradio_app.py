"""
Gradio chat interface for DataShare NLP.
"""
import gradio as gr

from src.database import init_db
from src.intent_parser import IntentParser
from src.semantic_search import SemanticSearch
from src.synthesizer import DatasetSynthesizer

intent_parser = IntentParser()
searcher = SemanticSearch()
synthesizer = DatasetSynthesizer()

MAX_TURNS = 5
CONTEXT_TURNS = 2
TRIAL_LIMIT_MESSAGE = (
    "**This is a trial version with a limited chat length.** "
    "Please click **Clear** (or refresh the page) to start a new session and keep exploring."
)


def chat_response(message: str, history: list) -> str:
    """
    Multi-turn conversation handler.

    Flow:
    1. Enforce a trial session length cap
    2. Parse intent from user message, using the last couple of turns as
       context so a follow-up ("after 2010") can be merged with an earlier
       data type/region ("fire history for Victoria")
    3. Check if clarification needed
    4. Search for relevant datasets
    5. Render the deterministic dataset-card template
    """
    completed_turns = len(history) // 2
    if completed_turns >= MAX_TURNS:
        return TRIAL_LIMIT_MESSAGE

    recent_user_messages = [
        turn["content"] for turn in history
        if turn.get("role") == "user" and isinstance(turn.get("content"), str)
    ][-CONTEXT_TURNS:]

    intent = intent_parser.parse_intent(message, recent_user_messages)

    if intent.get('clarification_needed'):
        return intent['clarification_question']

    # Embed a query that includes recent context, not just the current
    # message — otherwise a follow-up like "after 2010" embeds as an
    # isolated fragment and the vector search never retrieves datasets
    # about the subject (e.g. fire) established in the prior turn, even
    # though the intent parser correctly merged it into structured fields.
    search_query = " ".join(recent_user_messages + [message])

    retrieved_datasets = searcher.search(search_query, intent, limit=5)

    return synthesizer.format_as_template(message, retrieved_datasets)


def create_interface():
    demo = gr.ChatInterface(
        fn=chat_response,
        title="DEECA Data Discovery Assistant",
        description=(
            "Ask for environmental data in plain English. I'll help you find DEECA's datasets.\n\n"
            "**How to use:** Type your question in the box below, then either press **Enter** "
            "or click the send button (paper plane icon) to submit it. Each answer takes a few "
            "seconds to generate — wait for the response before sending another message.\n\n"
            "**Examples:**\n"
            "- \"Find biodiversity data for the Dandenongs\"\n"
            "- \"Show me water quality datasets for the Murray region\"\n"
            "- \"What fire history data do we have for Victoria?\"\n"
            "- \"I need current water monitoring data\""
        ),
        textbox=gr.Textbox(
            placeholder="E.g., 'I need water quality data for Gippsland' — then press Enter",
            lines=1,
        ),
    )
    return demo


if __name__ == "__main__":
    print("Initializing database...")
    init_db()

    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
