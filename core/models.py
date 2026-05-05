from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class ArticleIdea:
    title: str
    angle: str
    category: str
    local_relevance_score: float
    sources: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    priority: str = "media"
    confidence_score: float | None = None
    verdict: str | None = None

    def to_dict(self) -> dict:
        return self.__dict__

@dataclass
class ResearchReport:
    query: str
    trending_topics: list[str]
    article_ideas: list[ArticleIdea]
    context_snippets: list[str]
    raw_web_results: list[dict]

    def to_dict(self) -> dict:
        return {
            "query":            self.query,
            "trending_topics":  self.trending_topics,
            "article_ideas":    [a.to_dict() for a in self.article_ideas],
            "context_snippets": self.context_snippets,
        }

@dataclass
class CreateArticle:
    title: str
    angle: str
    category: str
    local_relevance_score: float
    article_content: str
    sources: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class FactCheckResult:
    idea: "ArticleIdea"
    verdict: str
    reason: str
    confidence: float
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "idea":       self.idea.to_dict(),
            "verdict":    self.verdict,
            "reason":     self.reason,
            "confidence": self.confidence,
            "sources":    self.sources,
        }

@dataclass
class VerificationResult:
    input_text: str
    verdict: str
    reason: str
    confidence: float
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__
