"""
Fetch DEECA datasets from Victoria's CKAN API
"""
from datetime import datetime
from typing import List, Dict

import requests

from src.database import get_session, Dataset, init_db
from src.embeddings import generate_embedding

CKAN_API_BASE = "https://discover.data.vic.gov.au/api/3/action"
DEECA_ORG_SLUG = "department-of-energy-environment-climate-action"


CKAN_PAGE_SIZE = 1000  # CKAN package_search caps `rows` at 1000 per request


def fetch_deeca_datasets(rows: int = 200) -> List[Dict]:
    """
    Fetch DEECA datasets from Victoria's CKAN API, paginating past CKAN's
    1000-rows-per-request cap if more than that are requested.

    Args:
        rows: Max number of datasets to fetch in total

    Returns:
        List of dataset dictionaries with metadata
    """
    url = f"{CKAN_API_BASE}/package_search"
    results: List[Dict] = []
    start = 0

    try:
        while len(results) < rows:
            page_size = min(CKAN_PAGE_SIZE, rows - len(results))
            params = {
                "q": f"organization:{DEECA_ORG_SLUG}",
                "rows": page_size,
                "start": start,
                "sort": "title_string asc"
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            if not data['success']:
                print("CKAN API error:", data)
                break

            page_results = data['result']['results']
            if not page_results:
                break  # no more datasets available

            results.extend(page_results)
            start += len(page_results)

            if len(page_results) < page_size:
                break  # last page was partial, nothing more to fetch

        return results

    except requests.exceptions.RequestException as e:
        print(f"Error fetching CKAN data: {e}")
        return results


def normalize_dataset(ckan_dataset: Dict) -> Dict:
    """
    Convert CKAN dataset format to our format.

    Args:
        ckan_dataset: Dataset from CKAN API

    Returns:
        Normalized dataset dictionary
    """
    return {
        'id': ckan_dataset.get('id', ''),
        'name': ckan_dataset.get('name', ''),
        'title': ckan_dataset.get('title', ''),
        'description': ckan_dataset.get('notes', ''),
        'owner': ckan_dataset.get('organization', {}).get('name', 'DEECA') if ckan_dataset.get('organization') else 'DEECA',
        'category': [tag['name'] for tag in ckan_dataset.get('tags', [])],
        'last_updated': ckan_dataset.get('metadata_modified', ''),
        'access_level': 'PUBLIC',  # CKAN doesn't always have this; infer or default
        'formats': extract_formats(ckan_dataset.get('resources', [])),
        'url': ckan_dataset.get('url') or f"https://discover.data.vic.gov.au/dataset/{ckan_dataset.get('name', '')}",
        'api_endpoint': None,
        'coverage': ['Victoria'],  # Assume Victoria-wide
        'update_frequency': 'Unknown',
        'data_types': []
    }


def extract_formats(resources: List[Dict]) -> List[str]:
    """Extract file formats from CKAN resources"""
    formats = set()
    for resource in resources:
        if resource.get('format'):
            formats.add(resource['format'].upper())
    return list(formats) if formats else ['CSV']


def populate_database(rows: int = 200):
    """
    Fetch datasets from CKAN API, generate embeddings, and populate database.

    Args:
        rows: Max number of datasets to fetch (paginates past CKAN's
            1000-per-request cap automatically). Pass a high number
            (e.g. 2000) to fetch the full DEECA catalog (~1220 datasets).
    """
    init_db()

    print("Fetching DEECA datasets from CKAN API...")
    ckan_datasets = fetch_deeca_datasets(rows=rows)

    if not ckan_datasets:
        print("No datasets found from CKAN API")
        return

    print(f"Found {len(ckan_datasets)} datasets. Populating database...")

    session = get_session()

    for ckan_ds in ckan_datasets:
        normalized = normalize_dataset(ckan_ds)

        # Generate embedding from title + description
        text_to_embed = f"{normalized['title']}. {normalized['description']}"
        embedding = generate_embedding(text_to_embed)

        # Parse last_updated date
        try:
            last_updated = datetime.fromisoformat(normalized['last_updated'].replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            last_updated = None

        dataset = Dataset(
            id=normalized['id'],
            name=normalized['name'],
            title=normalized['title'],
            description=normalized['description'],
            owner=normalized['owner'],
            category=normalized['category'],
            last_updated=last_updated,
            access_level=normalized['access_level'],
            formats=normalized['formats'],
            url=normalized['url'],
            api_endpoint=normalized['api_endpoint'],
            coverage=normalized['coverage'],
            update_frequency=normalized['update_frequency'],
            data_types=normalized['data_types'],
            description_embedding=embedding
        )

        session.merge(dataset)

    session.commit()
    session.close()

    print(f"Populated database with {len(ckan_datasets)} datasets")


if __name__ == "__main__":
    populate_database()
