import logging
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from .models import Category
from .session import get_db

logger = logging.getLogger(__name__)


def load_category_seeds(
    session: Session, csv_path: str = "seeds/categories.csv"
) -> None:
    """
    Load category seed data from a CSV file into the database.

    Reads category data from a CSV file and inserts it into the database
    using SQLAlchemy bulk insert.

    Args:
        session (Session): SQLAlchemy database session.
        csv_path (str, optional): Path to the CSV file containing category
            data. Defaults to "seeds/categories.csv".

    Raises:
        ValueError: If the CSV file format is incorrect.
        Exception: If any unexpected error occurs during data loading.
    """
    try:
        # Read CSV file with specified data types
        df = pd.read_csv(
            csv_path,
            dtype={
                "category_id": int,
                "category_name": str,
                "description": str,
                "answer_format": str,
                "group_id": int,
            },
        )

        # Prepare category objects for bulk insert
        categories = [
            Category(
                id=row["category_id"],
                name=row["category_name"],
                description=row["description"],
                answer_format=row["answer_format"],
                group_id=row["group_id"],
            )
            for _, row in df.iterrows()
        ]

        # Insert categories into the database
        session.bulk_save_objects(categories)
        session.commit()

        logger.info(f"Loaded {len(categories)} categories from seed file.")

    except pd.errors.ParserError as pe:
        logger.error(f"CSV parsing error: {pe}")
        raise ValueError("Invalid CSV file format.") from pe
    except Exception as e:
        logger.error(f"Unexpected error while loading categories: {e}")
        session.rollback()
        raise


def setup_database(seed_dir: str = "seeds", load_seeds: bool = True) -> None:
    """
    Initialize the database and optionally load seed data.

    This function checks for the existence of seed files and loads
    category data if available.

    Args:
        seed_dir (str, optional): Directory containing seed CSV files.
            Defaults to "seeds".
        load_seeds (bool, optional): Flag to indicate whether to load
            seed data. Defaults to True.
    """
    with get_db() as session:
        if load_seeds:
            categories_file = Path(seed_dir) / "category_descriptions.csv"
            if categories_file.exists():
                load_category_seeds(session, str(categories_file))
            else:
                logger.warning(f"Category seed file not found: {categories_file}")


if __name__ == "__main__":
    from .session import db_session

    # Initialize database tables
    db_session.create_database()

    # Load seed data
    setup_database()

    # Query database to verify loaded categories
    with get_db() as session:
        categories = session.query(Category).all()
        print("\nLoaded Categories:")
        for cat in categories:
            print(
                f"  - {cat.name} "
                f"(Group {cat.group_id if cat.group_id != 0 else 'None'})"
            )
