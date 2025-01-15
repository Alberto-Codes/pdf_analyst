from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    pass


class Category(Base):
    """
    Represents a CUAD (Contract Understanding Atticus Dataset) category.

    This model defines the contract categories used in the CUAD dataset,
    including their descriptions, expected answer formats, and grouping.

    Attributes:
        id (int): Unique identifier for the category (Primary Key).
        name (str): Name of the contract category (e.g., 'Document Name', 'Parties').
        description (str): Detailed description of the category's meaning.
        answer_format (str): Expected format of answers (e.g., 'Yes/No', 'Date').
        group_id (int): Category grouping identifier
            (0=ungrouped, 1=dates, 2=competition, 3=assignment,
            4=license, 5=post-term, 6=liability).
        created_at (datetime): Timestamp when the category was created.
    """

    __tablename__ = "categories"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Core fields
    name: Mapped[str] = mapped_column(
        String(100),  # Enforced length for DB optimization
        nullable=False,
        comment="Name of the contract category (e.g., 'Document Name', 'Parties')",
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Detailed description of what the category represents.",
    )

    answer_format: Mapped[str] = mapped_column(
        String(50),  # Enforced length constraint
        nullable=False,
        comment="Expected format of the answer (e.g., 'Yes/No', 'Date (mm/dd/yyyy)').",
    )

    group_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,  # Default to 0 if ungrouped
        comment="Category grouping (0=ungrouped, 1=dates, 2=competition, 3=assignment, "
        "4=license, 5=post-term, 6=liability).",
    )

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    def __repr__(self) -> str:
        """
        Returns a string representation of the Category instance.

        Returns:
            str: Readable representation of the category.
        """
        return f"Category(id={self.id}, name='{self.name}', group_id={self.group_id})"


class AnalysisResult(Base):
    """
    Represents the results of contract analysis.

    This model stores extracted data and AI predictions from contract
    documents, mapping them to predefined categories.

    Attributes:
        contract_path (str): Path to the analyzed contract file.
        category_id (int): Foreign key linking to a category.
        answer (str): Extracted answer related to the category.
        confidence (float): Confidence score of the extracted answer.
        extracted_text (Optional[str]): Additional extracted text (if available).
        analyzed_at (datetime): Timestamp of when the analysis was performed.
    """

    __tablename__ = "analysis_results"

    # Composite primary key (contract path and category)
    contract_path: Mapped[str] = mapped_column(
        String, primary_key=True, comment="Path to the analyzed contract file."
    )
    category_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("categories.id"),
        primary_key=True,
        comment="Foreign key referencing a category ID.",
    )

    # Analysis fields
    answer: Mapped[str] = mapped_column(
        Text, nullable=False, comment="Extracted answer related to the category."
    )

    confidence: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Confidence score of the extracted answer."
    )

    extracted_text: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="Additional extracted text, if available."
    )

    # Metadata
    analyzed_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        comment="Timestamp of when the analysis was performed.",
    )

    def __repr__(self) -> str:
        """
        Returns a string representation of the AnalysisResult instance.

        Returns:
            str: Readable representation of the analysis result.
        """
        return f"AnalysisResult(contract='{self.contract_path}', category={self.category_id})"
