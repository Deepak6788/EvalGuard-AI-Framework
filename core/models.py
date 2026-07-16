from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ClaimEvaluation:
    claim: str
    evidence: Optional[str]
    nli_label: str
    entailment_score: float
    contradiction_score: float
    neutral_score: float


@dataclass
class EvaluationResult:
    prompt: str
    response: str
    context: Optional[str]

    relevance_score: float = 0.0
    instruction_score: float = 0.0
    completeness_score: float = 0.0
    groundedness_score: float = 0.0
    hallucination_risk: float = 0.0

    overall_score: float = 0.0

    verdict: str = "NOT_EVALUATED"

    issues: List[str] = field(default_factory=list)

    claim_evaluations: List[ClaimEvaluation] = field(
        default_factory=list
    )