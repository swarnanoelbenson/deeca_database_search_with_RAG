"""
Generate natural language responses using retrieved datasets (RAG).
"""
from typing import Dict, List

from anthropic import Anthropic

from src.config import ANTHROPIC_API_KEY, MODEL

client = Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are a helpful data discovery assistant for DEECA (Victoria's Department of Energy, Environment and Climate Action).

Help users find and understand environmental datasets.

When presenting search results:
1. Explain which datasets match their query
2. Describe why each is relevant
3. Note any access restrictions
4. Highlight data freshness
5. Suggest next steps (how to access, format options, etc)

Be conversational, professional, and concise.
Keep explanations to 2-3 sentences per dataset."""


class DatasetSynthesizer:
    def __init__(self):
        self.model = MODEL

    def synthesize_response(
        self,
        user_query: str,
        retrieved_datasets: List[Dict],
        intent: Dict
    ) -> str:
        """
        Use Claude to synthesize a natural language response
        based on retrieved datasets (RAG pattern).
        """
        if not retrieved_datasets:
            return "I couldn't find datasets matching your request. Try being more specific about the region, data type, or access level needed."

        datasets_context = "Retrieved relevant datasets:\n\n"
        for i, ds in enumerate(retrieved_datasets, 1):
            datasets_context += f"{i}. **{ds['title']}**\n"
            datasets_context += f"   - Owner: {ds['owner']}\n"
            datasets_context += f"   - Description: {ds['description']}\n"
            datasets_context += f"   - Access Level: {ds['access_level']}\n"
            datasets_context += f"   - Available Formats: {', '.join(ds['formats'] or [])}\n"
            datasets_context += f"   - Coverage: {', '.join(ds['coverage'] or [])}\n"
            datasets_context += f"   - Last Updated: {ds['last_updated']}\n"
            datasets_context += f"   - URL: {ds['url']}\n\n"

        prompt = f"""User Query: {user_query}

{datasets_context}

Provide a helpful response explaining these datasets and why they match the user's request.
Include download/access information."""

        response = client.messages.create(
            model=self.model,
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        text_blocks = [block.text for block in response.content if block.type == "text"]
        if not text_blocks:
            return "I found matching datasets but couldn't generate a full explanation. Please see the list of datasets below."
        return "".join(text_blocks)

    def format_with_links(
        self,
        response_text: str,
        datasets: List[Dict]
    ) -> str:
        """Add dataset links to the synthesized response"""
        formatted = response_text

        if datasets:
            formatted += "\n\n**Found Datasets:**\n"
            for ds in datasets:
                formatted += f"\n- [{ds['title']}]({ds['url']})\n"
                formatted += f"  *Owner: {ds['owner']} | Access: {ds['access_level']} | Updated: {ds['last_updated']}*\n"

        return formatted

    def format_as_template(
        self,
        user_query: str,
        retrieved_datasets: List[Dict]
    ) -> str:
        """
        Render retrieved datasets as a deterministic template — no LLM call.

        Each dataset is a card: heading (linked), a metadata mini-table,
        the actual CKAN description as a blockquote, and a direct link.

        Note: no "year" is shown. CKAN only exposes a catalog last-refresh
        timestamp (metadata_modified), not the real data collection year —
        showing that as "the year" would misrepresent decade-old datasets
        (e.g. "2010 Index of Stream Condition") as current.
        """
        if not retrieved_datasets:
            return (
                f"## No datasets found for: \"{user_query}\"\n\n"
                "Try being more specific about the region, data type, or access level needed."
            )

        lines = [f"## Datasets matching: \"{user_query}\"\n"]

        for i, ds in enumerate(retrieved_datasets, 1):
            description = (ds['description'] or "No description available.").strip()
            quoted_description = "\n".join(f"> {line}" for line in description.splitlines())

            lines.append(f"### {i}. {ds['title']}\n")
            lines.append("| Field | Value |")
            lines.append("|---|---|")
            lines.append(f"| Access | {ds['access_level']} |")
            lines.append(f"| Formats | {', '.join(ds['formats'] or []) or 'Unknown'} |")
            lines.append(f"| Coverage | {', '.join(ds['coverage'] or []) or 'Unknown'} |")
            lines.append(f"| Catalog updated | {ds['last_updated'] or 'Unknown'} |")
            lines.append(f"| Relevance | {ds.get('relevance_score', 0)} |")
            lines.append("")
            lines.append(quoted_description)
            lines.append("")
            lines.append(f"[View dataset →]({ds['url']})")
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)
