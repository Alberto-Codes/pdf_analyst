import logging
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseSession:
    """Handles database session management."""

    def __init__(self, db_path: str = "pdf_analyst.duckdb"):
        """
        Initialize database session manager.

        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = Path(db_path)
        # Create duckdb-compatible SQLAlchemy engine
        self.engine = create_engine(f"duckdb:///{self.db_path}")
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def create_database(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Created database tables")

    def drop_database(self) -> None:
        """Drop all database tables."""
        Base.metadata.drop_all(bind=self.engine)
        logger.info("Dropped database tables")

    @contextmanager
    def session(self) -> Session:
        """Provide a transactional scope around a series of operations."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            logger.error(f"Session error: {e}")
            session.rollback()
            raise
        finally:
            session.close()


# Global session manager
db_session = DatabaseSession()


def get_session() -> DatabaseSession:
    """Get the database session manager."""
    return db_session


@contextmanager
def get_db() -> Session:
    """Context manager for database sessions."""
    with get_session().session() as session:
        yield session
