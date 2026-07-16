from evaluators.groundedness import GroundednessEvaluator


def main():

    evaluator = GroundednessEvaluator()

    context = (
        "Nova was created by Arin. "
        "The programming language was released in 2023."
    )

    response = (
        "Nova was created by Arin and released in 2023. "
        "It is widely used for machine learning."
    )

    result = evaluator.evaluate(
        response=response,
        context=context,
    )

    print("\nGROUNDEDNESS TEST")
    print("=" * 70)

    print(
        f"Groundedness Score: "
        f"{result['groundedness_score']}"
    )

    print()

    for index, claim in enumerate(
        result["claim_evaluations"],
        start=1,
    ):

        print(f"Claim {index}")
        print("-" * 70)

        print("Claim:")
        print(claim.claim)

        print("\nEvidence:")
        print(claim.evidence)

        print("\nLabel:")
        print(claim.nli_label)

        print(
            f"\nEntailment: "
            f"{claim.entailment_score:.4f}"
        )

        print(
            f"Neutral: "
            f"{claim.neutral_score:.4f}"
        )

        print(
            f"Contradiction: "
            f"{claim.contradiction_score:.4f}"
        )

        print()


if __name__ == "__main__":
    main()