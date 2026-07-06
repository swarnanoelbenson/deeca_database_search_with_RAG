import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/datashare_nlp")

# Model
MODEL = os.getenv("MODEL", "claude-sonnet-5")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# App
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
