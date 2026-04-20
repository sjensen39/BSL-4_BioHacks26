from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


ColumnType = Literal["numeric", "categorical", "datetime", "text", "boolean", "unknown"]


class ColumnProfile(BaseModel):
    name: str
    detected_type: ColumnType
    missing_count: int
    unique_count: int
    sample_values: List[str] = Field(default_factory=list)


class DataUploadResponse(BaseModel):
    data_id: str
    filename: str
    description: str
    row_count: int
    column_count: int
    columns: List[ColumnProfile]
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    datetime_columns: List[str] = Field(default_factory=list)
    preview_rows: List[dict] = Field(default_factory=list)
    overview_bullets: List[str] = Field(default_factory=list)


class AskRequest(BaseModel):
    data_id: str
    question: str = Field(min_length=3, max_length=1200)


class GraphRecommendation(BaseModel):
    chart_type: str
    title: str
    why_this_chart: str
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    group_column: Optional[str] = None
    steps: List[str] = Field(default_factory=list)
    python_code: str
    interpretation_notes: List[str] = Field(default_factory=list)
    good_when: Optional[str] = None


class AskResponse(BaseModel):
    direct_answer: str
    question_intent: str
    recommended_graphs: List[GraphRecommendation] = Field(default_factory=list)
    observations: List[str] = Field(default_factory=list)
    interpretation_help: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    matched_columns: List[str] = Field(default_factory=list)
    confidence_note: str = ""
