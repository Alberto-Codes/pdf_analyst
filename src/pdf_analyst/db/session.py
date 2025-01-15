import logging
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base  # Ensure models.py contains SQLAlchemy models

logger = logging.getLogger(__name__)


class DatabaseSession:
    """Manages SQLite database connections and sessions."""

    def __init__(self, db_path: str = None):
        """
        Initializes the database session manager.

        Args:
            db_path (str, optional): Path to the SQLite database file.
                Defaults to "data/pdf_analyst.sqlite" inside the repository root.
        """
        # Define default database directory
        default_dir = Path("data")
        default_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        # Use provided path or default to 'data/pdf_analyst.sqlite'
        self.db_path = Path(db_path) if db_path else default_dir / "pdf_analyst.sqlite"
        self.db_path = self.db_path.resolve()  # Ensure absolute path resolution

        # Create the SQLAlchemy engine
        self.engine = create_engine(f"sqlite:///{self.db_path}")

        # Configure session factory
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def create_database(self) -> None:
        """
        Creates all tables defined in the SQLAlchemy models.

        This function initializes the database schema by creating tables
        based on the ORM models.

        Raises:
            Exception: If table creation fails.
        """
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info(f"Successfully created database tables at {self.db_path}")
        except Exception as e:
            logger.exception(f"Failed to create database tables: {e}")
            raise

    def drop_database(self) -> None:
        """
        Drops all database tables.

        This function removes all existing tables from the database.

        Raises:
            Exception: If dropping tables fails.
        """
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Successfully dropped all database tables.")
        except Exception as e:
            logger.exception(f"Failed to drop database tables: {e}")
            raise

    @contextmanager
    def session(self) -> Session:
        """
        Provides a transactional scope for database operations.

        This function ensures that a session is created, used, and closed
        properly while handling transactions.

        Yields:
            Session: SQLAlchemy session object.

        Raises:
            Exception: If a database transaction fails.
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            logger.exception("Database session encountered an error.")
            session.rollback()
            raise
        finally:
            session.close()


# Global database session instance with 'data/' directory enforced
db_session = DatabaseSession()


def get_session() -> DatabaseSession:
    """
    Returns the global database session manager instance.

    This function ensures a single instance of the database session
    manager is used throughout the application.

    Returns:
        DatabaseSession: An instance of the database session manager.
    """
    return db_session


@contextmanager
def get_db() -> Session:
    """
    Provides a database session for transactional operations.

    This function wraps `DatabaseSession.session()` to simplify session
    handling when interacting with the database.

    Yields:
        Session: SQLAlchemy session object.
    """
    with get_session().session() as session:
        yield session
