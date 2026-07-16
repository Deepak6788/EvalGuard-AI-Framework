from typing import List

from core.models import ClaimEvaluation


class HallucinationEvaluator:
    CONTRADICTION_WEIGHT = 1.0
    NEUTRAL_WEIGHT = 0.5
    ENTAILMENT_WEIGHT = 0.0

    def evaluate(
        self,
        claim_evaluations: List[ClaimEvaluation],
    ) -> dict:
        if not claim_evaluations:
            return {
                "hallucination_risk": 0.0,
                "risk_label": "NONE",
                "contradicted_claims": 0,
                "unsupported_claims": 0,
                "supported_claims": 0,
            }

        total_risk = 0.0

        contradicted_claims = 0
        unsupported_claims = 0
        supported_claims = 0

        for evaluation in claim_evaluations:
            if evaluation.nli_label == "CONTRADICTION":
                total_risk += self.CONTRADICTION_WEIGHT
                contradicted_claims += 1

            elif evaluation.nli_label == "NEUTRAL":
                total_risk += self.NEUTRAL_WEIGHT
                unsupported_claims += 1

            elif evaluation.nli_label == "ENTAILMENT":
                total_risk += self.ENTAILMENT_WEIGHT
                supported_claims += 1

        hallucination_risk = (
            total_risk / len(claim_evaluations)
        )

        risk_label = self._get_risk_label(
            hallucination_risk
        )

        return {
            "hallucination_risk": round(
                hallucination_risk,
                4,
            ),
            "risk_label": risk_label,
            "contradicted_claims": contradicted_claims,
            "unsupported_claims": unsupported_claims,
            "supported_claims": supported_claims,
        }

    @staticmethod
    def _get_risk_label(
        hallucination_risk: float,
    ) -> str:
        if hallucination_risk >= 0.75:
            return "CRITICAL"

        if hallucination_risk >= 0.50:
            return "HIGH"

        if hallucination_risk >= 0.25:
            return "MEDIUM"

        if hallucination_risk > 0.0:
            return "LOW"

        return "NONE"