from core.models import EvaluationResult
from evaluators.completeness import CompletenessEvaluator
from evaluators.groundedness import GroundednessEvaluator
from evaluators.hallucination import HallucinationEvaluator
from evaluators.instruction import InstructionEvaluator
from evaluators.relevance import RelevanceEvaluator


class EvaluationEngine:
    RELEVANCE_WEIGHT = 0.20
    INSTRUCTION_WEIGHT = 0.15
    COMPLETENESS_WEIGHT = 0.20
    GROUNDEDNESS_WEIGHT = 0.30
    HALLUCINATION_WEIGHT = 0.15

    def __init__(self):
        print("Initializing EvalGuard evaluation engine...")

        self.relevance_evaluator = RelevanceEvaluator()

        self.instruction_evaluator = (
            InstructionEvaluator()
        )

        self.completeness_evaluator = (
            CompletenessEvaluator(
                relevance_evaluator=self.relevance_evaluator
            )
        )

        self.groundedness_evaluator = (
            GroundednessEvaluator()
        )

        self.hallucination_evaluator = (
            HallucinationEvaluator()
        )

        print("EvalGuard evaluation engine ready.")

    def evaluate(
        self,
        prompt: str,
        response: str,
        context: str,
    ) -> EvaluationResult:

        relevance_result = (
            self.relevance_evaluator.evaluate(
                prompt=prompt,
                response=response,
            )
        )

        instruction_result = (
            self.instruction_evaluator.evaluate(
                prompt=prompt,
                response=response,
            )
        )

        completeness_result = (
            self.completeness_evaluator.evaluate(
                prompt=prompt,
                response=response,
            )
        )

        groundedness_result = (
            self.groundedness_evaluator.evaluate(
                response=response,
                context=context,
            )
        )

        hallucination_result = (
            self.hallucination_evaluator.evaluate(
                claim_evaluations=
                groundedness_result[
                    "claim_evaluations"
                ]
            )
        )

        relevance_score = (
            relevance_result["score"]
        )

        instruction_score = (
            instruction_result[
                "instruction_score"
            ]
        )

        completeness_score = (
            completeness_result[
                "completeness_score"
            ]
        )

        groundedness_score = (
            groundedness_result[
                "groundedness_score"
            ]
        )

        hallucination_risk = (
            hallucination_result[
                "hallucination_risk"
            ]
        )

        overall_score = (
            self._calculate_overall_score(
                relevance_score=relevance_score,
                instruction_score=instruction_score,
                completeness_score=completeness_score,
                groundedness_score=groundedness_score,
                hallucination_risk=hallucination_risk,
            )
        )

        verdict = self._get_verdict(
            overall_score=overall_score,
            hallucination_risk=hallucination_risk,
        )

        issues = self._collect_issues(
            instruction_result=instruction_result,
            hallucination_result=
            hallucination_result,
        )

        return EvaluationResult(
            prompt=prompt,
            response=response,
            context=context,

            relevance_score=relevance_score,
            instruction_score=instruction_score,
            completeness_score=completeness_score,
            groundedness_score=groundedness_score,
            hallucination_risk=hallucination_risk,

            overall_score=overall_score,

            verdict=verdict,

            issues=issues,

            claim_evaluations=
            groundedness_result[
                "claim_evaluations"
            ],
        )
    def _calculate_overall_score(
        self,
        relevance_score: float,
        instruction_score: float,
        completeness_score: float,
        groundedness_score: float,
        hallucination_risk: float,
    ) -> float:

        hallucination_safety_score = (
            1.0 - hallucination_risk
        )

        overall_score = (
            relevance_score * self.RELEVANCE_WEIGHT
            + instruction_score * self.INSTRUCTION_WEIGHT
            + completeness_score * self.COMPLETENESS_WEIGHT
            + groundedness_score * self.GROUNDEDNESS_WEIGHT
            + hallucination_safety_score
            * self.HALLUCINATION_WEIGHT
        )

        return round(overall_score, 4)

    @staticmethod
    def _get_verdict(
        overall_score: float,
        hallucination_risk: float,
    ) -> str:

        if hallucination_risk >= 0.75:
            return "UNSAFE"

        if overall_score >= 0.90:
            return "EXCELLENT"

        if overall_score >= 0.75:
            return "GOOD"

        if overall_score >= 0.55:
            return "NEEDS_REVIEW"

        return "POOR"

    @staticmethod
    def _collect_issues(
        instruction_result: dict,
        hallucination_result: dict,
    ) -> list:

        issues = list(
            instruction_result["violations"]
        )

        contradicted_claims = (
            hallucination_result[
                "contradicted_claims"
            ]
        )

        unsupported_claims = (
            hallucination_result[
                "unsupported_claims"
            ]
        )

        if contradicted_claims > 0:
            issues.append(
                f"{contradicted_claims} claim(s) "
                "contradict the reference context."
            )

        if unsupported_claims > 0:
            issues.append(
                f"{unsupported_claims} claim(s) "
                "are unsupported by the reference context."
            )

        return issues