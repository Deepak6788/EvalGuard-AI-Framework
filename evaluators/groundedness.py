import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from config import (
    CONTRADICTION_THRESHOLD,
    ENTAILMENT_THRESHOLD,
    EVIDENCE_SIMILARITY_THRESHOLD,
    NLI_MODEL_NAME,
)
from core.evidence_retriever import EvidenceRetriever
from core.models import ClaimEvaluation
from utils.text_processing import extract_claims


class NLIAnalyzer:
    def __init__(self):
        print("Loading NLI model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            NLI_MODEL_NAME
        )

        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                NLI_MODEL_NAME
            )
        )

        self.model.eval()

        print("NLI model loaded.")

    def analyze(
        self,
        evidence: str,
        claim: str,
    ) -> dict:
        inputs = self.tokenizer(
            evidence,
            claim,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(
            outputs.logits[0],
            dim=-1,
        )

        scores = {
            "ENTAILMENT": probabilities[0].item(),
            "NEUTRAL": probabilities[1].item(),
            "CONTRADICTION": probabilities[2].item(),
        }

        predicted_label = max(
            scores,
            key=scores.get,
        )

        return {
            "label": predicted_label,
            "entailment_score": round(
                scores["ENTAILMENT"],
                4,
            ),
            "neutral_score": round(
                scores["NEUTRAL"],
                4,
            ),
            "contradiction_score": round(
                scores["CONTRADICTION"],
                4,
            ),
        }


class GroundednessEvaluator:
    def __init__(self):
        self.evidence_retriever = EvidenceRetriever()
        self.nli_analyzer = NLIAnalyzer()

    def evaluate(
        self,
        response: str,
        context: str,
    ) -> dict:
        claims = extract_claims(response)

        if not claims:
            return {
                "groundedness_score": 0.0,
                "claim_evaluations": [],
            }

        claim_evaluations = []

        for claim in claims:
            evidence_candidates = (
                self.evidence_retriever.retrieve(
                    claim=claim,
                    context=context,
                    top_k=5,
                )
            )

            claim_evaluation = self._evaluate_claim(
                claim=claim,
                evidence_candidates=evidence_candidates,
            )

            claim_evaluations.append(claim_evaluation)

        supported_claims = sum(
            1
            for evaluation in claim_evaluations
            if evaluation.nli_label == "ENTAILMENT"
        )

        groundedness_score = (
            supported_claims / len(claim_evaluations)
        )

        return {
            "groundedness_score": round(
                groundedness_score,
                4,
            ),
            "claim_evaluations": claim_evaluations,
        }

    def _evaluate_claim(
        self,
        claim: str,
        evidence_candidates: list,
    ) -> ClaimEvaluation:
        valid_candidates = [
            candidate
            for candidate in evidence_candidates
            if candidate["similarity_score"]
            >= EVIDENCE_SIMILARITY_THRESHOLD
        ]

        if not valid_candidates:
            return ClaimEvaluation(
                claim=claim,
                evidence=None,
                nli_label="NEUTRAL",
                entailment_score=0.0,
                contradiction_score=0.0,
                neutral_score=1.0,
            )

        analyzed_candidates = []

        for candidate in valid_candidates:
            nli_result = self.nli_analyzer.analyze(
                evidence=candidate["evidence"],
                claim=claim,
            )

            analyzed_candidates.append(
                {
                    "evidence": candidate["evidence"],
                    "similarity_score": candidate[
                        "similarity_score"
                    ],
                    **nli_result,
                }
            )

        best_candidate = self._select_best_candidate(
            analyzed_candidates
        )

        return ClaimEvaluation(
            claim=claim,
            evidence=best_candidate["evidence"],
            nli_label=best_candidate["label"],
            entailment_score=best_candidate[
                "entailment_score"
            ],
            contradiction_score=best_candidate[
                "contradiction_score"
            ],
            neutral_score=best_candidate[
                "neutral_score"
            ],
        )

    @staticmethod
    def _select_best_candidate(
        analyzed_candidates: list,
    ) -> dict:
        strong_entailments = [
            candidate
            for candidate in analyzed_candidates
            if candidate["entailment_score"]
            >= ENTAILMENT_THRESHOLD
        ]

        strong_contradictions = [
            candidate
            for candidate in analyzed_candidates
            if candidate["contradiction_score"]
            >= CONTRADICTION_THRESHOLD
        ]

        strongest_entailment = (
            max(
                strong_entailments,
                key=lambda candidate: candidate[
                    "entailment_score"
                ],
            )
            if strong_entailments
            else None
        )

        strongest_contradiction = (
            max(
                strong_contradictions,
                key=lambda candidate: candidate[
                    "contradiction_score"
                ],
            )
            if strong_contradictions
            else None
        )

        if (
            strongest_entailment is not None
            and strongest_contradiction is not None
        ):
            if (
                strongest_contradiction[
                    "contradiction_score"
                ]
                > strongest_entailment[
                    "entailment_score"
                ]
            ):
                return strongest_contradiction

            return strongest_entailment

        if strongest_contradiction is not None:
            return strongest_contradiction

        if strongest_entailment is not None:
            return strongest_entailment

        return max(
            analyzed_candidates,
            key=lambda candidate: candidate[
                "neutral_score"
            ],
        )