import logging
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from pdf_analyst.db.models import Category
from pdf_analyst.db.session import get_db

logger = logging.getLogger(__name__)


def load_category_seeds(
    session: Session, csv_path: str = "seeds/category_descriptions.csv"
) -> None:
    """
    Load categories from seed CSV file.

    Args:
        session: SQLAlchemy session
        csv_path: Path to categories CSV
    """
    try:
        # Read CSV
        df = pd.read_csv(csv_path)

        # Transform data
        categories = []
        for i, row in df.iterrows():
            category = Category(
                id=i + 1,
                name=row["Category (incl. context and answer)"].replace(
                    "Category: ", ""
                ),
                description=row["Description"],
                answer_format=row["Answer Format"],
                group_id=-1 if row["Group"] == "-" else int(row["Group"]),
            )
            categories.append(category)

        # Bulk insert
        session.bulk_save_objects(categories)
        session.commit()

        logger.info(f"Loaded {len(categories)} categories from seed file")

    except Exception as e:
        logger.error(f"Error loading categories from seed: {e}")
        raise


def setup_database(seed_dir: str = "seeds", load_seeds: bool = True) -> None:
    """
    Set up database with optional seed data loading.

    Args:
        seed_dir: Directory containing seed files
        load_seeds: Whether to load seed data
    """
    with get_db() as session:
        # Load category seeds if requested
        if load_seeds:
            categories_file = Path(seed_dir) / "category_descriptions.csv"
            if categories_file.exists():
                load_category_seeds(session, str(categories_file))
            else:
                logger.warning(f"Categories file not found at {categories_file}")


if __name__ == "__main__":
    # Example usage
    from .session import db_session

    # Create tables
    db_session.create_database()

    # Load seeds
    setup_database()

    # Query example
    with get_db() as session:
        categories = session.query(Category).all()
        print("\nLoaded Categories:")
        for cat in categories:
            print(f"  - {cat.name} (Group {cat.group_id})")
