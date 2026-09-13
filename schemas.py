"""
Pydantic v2 data models shared across UdaPlay's tools and agents.

Keeping every structured payload (game records, evaluation results,
and the final answer) as a validated Pydantic model is what lets the
orchestrator pass data between agents safely and lets the "overall"
tool guarantee a consistent, well-formed final response.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class GameRecord(BaseModel):
    """A single video game entry as stored in data/games.json / FAISS."""

    Name: str
    Platform: str
    Genre: str
    Publisher: str
    Description: str
    ReleaseYear: int


class RetrievalEvaluation(BaseModel):
    """Structured output of the validation agent / evaluate_retrieval tool."""

    confidence_score: float = Field(
        ge=0.0, le=1.0, description="How confident we are the retrieved context answers the query."
    )
    is_sufficient: bool = Field(
        description="Whether the retrieved internal context is sufficient to answer the query without a web search."
    )
    reasoning: str = Field(description="Short explanation for the confidence score and sufficiency decision.")


SourceType = Literal[
    "Internal Game Database",
    "Web Search via Tavily",
    "Internal Game Database + Web Search via Tavily",
]


class KeyDetails(BaseModel):
    """Highlighted fields every final answer should surface when known."""

    platform: Optional[str] = None
    publisher: Optional[str] = None
    release_date: Optional[str] = None


class FinalAnswer(BaseModel):
    """The structured final answer produced by the overall/synthesizer tool."""

    query: str
    answer: str
    key_details: KeyDetails
    source: SourceType
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    def to_display_string(self) -> str:
        """Human-readable rendering used by the CLI / notebooks."""
        lines = [self.answer, "", "Key Details:"]
        if self.key_details.platform:
            lines.append(f"  - Platform: {self.key_details.platform}")
        if self.key_details.publisher:
            lines.append(f"  - Publisher: {self.key_details.publisher}")
        if self.key_details.release_date:
            lines.append(f"  - Release Date: {self.key_details.release_date}")
        lines.append("")
        lines.append(f"Source: {self.source}")
        if self.confidence is not None:
            lines.append(f"(Retrieval confidence: {self.confidence:.2f})")
        return "\n".join(lines)
