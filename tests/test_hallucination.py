from evaluators.hallucination import HallucinationEvaluator
from core.models import ClaimEvaluation


def main():

    evaluator = HallucinationEvaluator()

    test_cases = [

        {
            "name": "All Supported",
            "claims": [
                ClaimEvaluation(
                    claim="Nova was created by Arin.",
                    evidence="Nova was created by Arin.",
                    nli_label="ENTAILMENT",
                    entailment_score=0.99,
                    contradiction_score=0.00,
                    neutral_score=0.01,
                )
            ],
        },

        {
            "name": "One Unsupported",
            "claims": [
                ClaimEvaluation(
                    claim="Nova was created by Arin.",
                    evidence="Nova was created by Arin.",
                    nli_label="ENTAILMENT",
                    entailment_score=0.99,
                    contradiction_score=0.00,
                    neutral_score=0.01,
                ),
                ClaimEvaluation(
                    claim="Nova is used for ML.",
                    evidence=None,
                    nli_label="NEUTRAL",
                    entailment_score=0.00,
                    contradiction_score=0.00,
                    neutral_score=1.00,
                ),
            ],
        },

        {
            "name": "Contradiction",
            "claims": [
                ClaimEvaluation(
                    claim="Nova was created by Zoren.",
                    evidence="Nova was created by Arin.",
                    nli_label="CONTRADICTION",
                    entailment_score=0.00,
                    contradiction_score=0.99,
                    neutral_score=0.01,
                )
            ],
        },

    ]

    print("\nHALLUCINATION TEST")
    print("=" * 70)

    for test in test_cases:

        result = evaluator.evaluate(
            claim_evaluations=test["claims"]
        )

        print(f"\n{test['name']}")
        print("-" * 70)

        print(
            "Hallucination Risk:",
            result["hallucination_risk"],
        )

        print(
            "Risk Label:",
            result["risk_label"],
        )

        print(
            "Supported Claims:",
            result["supported_claims"],
        )

        print(
            "Unsupported Claims:",
            result["unsupported_claims"],
        )

        print(
            "Contradicted Claims:",
            result["contradicted_claims"],
        )


if __name__ == "__main__":
    main()