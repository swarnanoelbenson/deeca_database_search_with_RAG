"""
Gradio chat interface for DataShare NLP.
"""
import json

import gradio as gr

from src.database import init_db
from src.intent_parser import IntentParser, MAX_HISTORY_TURNS
from src.semantic_search import SemanticSearch
from src.synthesizer import DatasetSynthesizer

intent_parser = IntentParser()
searcher = SemanticSearch()
synthesizer = DatasetSynthesizer()

WELCOME_MESSAGE = (
    "Hi! I'm the **DEECA Data Discovery Assistant**. Ask for environmental data in plain "
    "English — e.g. \"Find biodiversity data for the Dandenongs\" — and click **Search** "
    "(or press Enter).\n\n"
    "**Refining results:** this is a multi-turn session — after your first search, ask a "
    "follow-up (e.g. \"in Victoria, after 2010\") and click **Search** again. I'll merge it "
    "with what you asked for in your last search rather than starting over.\n\n"
    "**How this works:** this is a RAG (Retrieval-Augmented Generation) pipeline. Claude "
    "first parses your request into structured intent (data type, region, constraints), "
    "then a semantic search step embeds it and ranks datasets by cosine similarity over "
    "vector embeddings (pgvector) blended with keyword/category matches, and finally the "
    "top matches are rendered as structured dataset cards — no free-form generation of "
    "dataset facts, so what you see is always grounded in real catalog data.\n\n"
    "**Scope note:** this is searching DEECA's full catalog of **~1220 datasets**.\n\n"
    "Click **Clear** at any point to start a fresh session."
)

CUSTOM_CSS = """
#search-btn, #clear-btn {
    color: #ffffff !important;
    border: none !important;
    font-weight: 600;
}
#search-btn { background-color: #16a34a !important; }
#search-btn:hover { background-color: #15803d !important; }
#clear-btn { background-color: #dc2626 !important; }
#clear-btn:hover { background-color: #b91c1c !important; }
"""


def chat_response(message: str, intent_history: list, intent: dict) -> str:
    """Render the search results for an already-parsed intent."""
    if intent.get('clarification_needed'):
        return intent['clarification_question']

    # Embed a query that includes recent user turns, not just the current
    # message — otherwise a follow-up embeds as an isolated fragment and
    # the vector search never retrieves datasets about the subject
    # established in a prior turn, even though the intent parser correctly
    # merged it into structured fields.
    recent_user_messages = [
        turn["content"] for turn in intent_history if turn.get("role") == "user"
    ]
    search_query = " ".join(recent_user_messages + [message])

    retrieved_datasets = searcher.search(search_query, intent, limit=5)

    return synthesizer.format_as_template(message, retrieved_datasets)


def respond(message: str, history: list, intent_history: list):
    """
    Submit handler: parse intent using real prior turns (not a paraphrased
    summary), run the search, and update both the display chat history and
    the bounded raw-JSON intent transcript.
    """
    history = history or []
    intent_history = intent_history or []

    if not message or not message.strip():
        return "", history, intent_history

    intent = intent_parser.parse_intent(message, intent_history)
    reply = chat_response(message, intent_history, intent)

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]

    # Store Claude's real JSON reply (not the rendered markdown cards) as
    # the assistant turn, bounded to the last MAX_HISTORY_TURNS turns, so
    # the next call gives Claude its own actual prior output as context.
    intent_history = intent_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": json.dumps(intent)},
    ]
    intent_history = intent_history[-(MAX_HISTORY_TURNS * 2):]

    return "", history, intent_history


def reset_session():
    return (
        [{"role": "assistant", "content": WELCOME_MESSAGE}],  # chatbot
        [],  # intent_history
        "",  # message textbox
    )


def create_interface():
    with gr.Blocks(title="DEECA Data Discovery Assistant") as demo:
        gr.Markdown("# DEECA Data Discovery Assistant")

        chatbot = gr.Chatbot(
            label="Conversation",
            height=650,
            value=[{"role": "assistant", "content": WELCOME_MESSAGE}],
        )
        intent_history_state = gr.State([])

        with gr.Row():
            msg = gr.Textbox(
                placeholder="E.g., 'I need water quality data for Gippsland' — then press Enter",
                lines=1,
                scale=8,
                show_label=False,
                container=False,
            )
            search_btn = gr.Button("Search", scale=1, elem_id="search-btn")
            clear_btn = gr.Button("Clear", scale=1, elem_id="clear-btn")

        search_inputs = [msg, chatbot, intent_history_state]
        search_outputs = [msg, chatbot, intent_history_state]

        msg.submit(respond, search_inputs, search_outputs)
        search_btn.click(respond, search_inputs, search_outputs)
        clear_btn.click(reset_session, None, [chatbot, intent_history_state, msg])

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
        css=CUSTOM_CSS,
    )
