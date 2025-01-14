from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all models"""

    pass


class Category(Base):
    """CUAD category definition."""

    __tablename__ = "categories"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Core fields
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    answer_format: Mapped[str] = mapped_column(String, nullable=False)
    group_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    def __repr__(self) -> str:
        return f"Category(id={self.id}, name='{self.name}', group_id={self.group_id})"


class AnalysisResult(Base):
    """Results of contract analysis."""

    __tablename__ = "analysis_results"

    # Composite primary key
    contract_path: Mapped[str] = mapped_column(String, primary_key=True)
    category_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("categories.id"), primary_key=True
    )

    # Analysis fields
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    extracted_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metadata
    analyzed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    def __repr__(self) -> str:
        return f"AnalysisResult(contract='{self.contract_path}', category={self.category_id})"
