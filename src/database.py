from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from datetime import datetime
from src.config import DATABASE_URL

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True)
    name = Column(String, unique=True, index=True)
    title = Column(String)
    description = Column(Text)
    owner = Column(String)
    category = Column(ARRAY(String), index=True)
    last_updated = Column(DateTime)
    access_level = Column(String)  # PUBLIC, OFFICIAL, PROTECTED, SECRET
    formats = Column(ARRAY(String))
    url = Column(String)
    api_endpoint = Column(String, nullable=True)
    coverage = Column(ARRAY(String))
    update_frequency = Column(String)
    data_types = Column(ARRAY(String))

    # Vector embedding of description + title
    description_embedding = Column(Vector(1536))

    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'title': self.title,
            'description': self.description,
            'owner': self.owner,
            'category': self.category,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'access_level': self.access_level,
            'formats': self.formats,
            'url': self.url,
            'coverage': self.coverage,
            'update_frequency': self.update_frequency,
            'data_types': self.data_types
        }


def init_db():
    """Create all tables and the pgvector extension if missing."""
    with engine.connect() as conn:
        conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
    Base.metadata.create_all(engine)


def get_session():
    return SessionLocal()
