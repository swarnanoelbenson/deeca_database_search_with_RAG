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


def chat_response(message: str, history: list) -> str:
    """
    Multi-turn conversation handler.

    Flow:
    1. Parse intent from user message
    2. Check if clarification needed
    3. Search for relevant datasets
    4. Synthesize natural language response
    5. Format with links
    """
    intent = intent_parser.parse_intent(message)

    if intent.get('clarification_needed'):
        return intent['clarification_question']

    retrieved_datasets = searcher.search(message, intent, limit=5)

    response_text = synthesizer.synthesize_response(
        message,
        retrieved_datasets,
        intent
    )

    return synthesizer.format_with_links(response_text, retrieved_datasets)


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
