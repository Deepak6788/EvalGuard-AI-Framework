from core.models import EvaluationResult


class ReportFormatter:

    @staticmethod
    def print_report(
        result: EvaluationResult,
        execution_time: float,
    ):

        print()
        print("=" * 70)
        print("                     EVALGUARD REPORT")
        print("=" * 70)

        print(f"Verdict              : {result.verdict}")
        print(f"Overall Score        : {result.overall_score:.4f}")
        print(
            f"Execution Time       : "
            f"{execution_time:.2f} seconds"
        )

        print()

        print("Evaluation Metrics")
        print("-" * 70)

        print(
            f"Relevance            : "
            f"{result.relevance_score:.4f}"
        )

        print(
            f"Instruction          : "
            f"{result.instruction_score:.4f}"
        )

        print(
            f"Completeness         : "
            f"{result.completeness_score:.4f}"
        )

        print(
            f"Groundedness         : "
            f"{result.groundedness_score:.4f}"
        )

        print(
            f"Hallucination Risk   : "
            f"{result.hallucination_risk:.4f}"
        )

        print()

        print("Detected Issues")
        print("-" * 70)

        if result.issues:
            for i, issue in enumerate(
                result.issues,
                start=1,
            ):
                print(f"{i}. {issue}")
        else:
            print("None")

        print()

        print("Claim-Level Analysis")
        print("-" * 70)

        if not result.claim_evaluations:
            print("No claims extracted.")

        else:

            for index, claim in enumerate(
                result.claim_evaluations,
                start=1,
            ):

                print()

                print(f"Claim {index}")
                print(f"Claim: {claim.claim}")
                print(f"Evidence: {claim.evidence}")
                print(f"Label: {claim.nli_label}")

                print(
                    f"Entailment: "
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

        ReportFormatter.print_summary(result)

    @staticmethod
    def print_summary(
        result: EvaluationResult,
    ):

        print()
        print("=" * 70)
        print("                 EVALUATION SUMMARY")
        print("=" * 70)

        strengths = []
        weaknesses = []

        if result.relevance_score >= 0.75:
            strengths.append(
                "Highly relevant to the prompt."
            )
        else:
            weaknesses.append(
                "Low response relevance."
            )

        if result.instruction_score >= 0.90:
            strengths.append(
                "Instructions followed correctly."
            )
        else:
            weaknesses.append(
                "Instruction-following issues detected."
            )

        if result.completeness_score >= 0.80:
            strengths.append(
                "Prompt answered completely."
            )
        else:
            weaknesses.append(
                "Prompt not answered completely."
            )

        if result.groundedness_score >= 0.80:
            strengths.append(
                "Claims are well supported."
            )
        else:
            weaknesses.append(
                "Some claims lack supporting evidence."
            )

        if result.hallucination_risk <= 0.25:
            strengths.append(
                "Low hallucination risk."
            )
        else:
            weaknesses.append(
                "Potential hallucinated content detected."
            )

        print("\nStrengths")

        if strengths:
            for item in strengths:
                print(f"✓ {item}")
        else:
            print("None")

        print("\nWeaknesses")

        if weaknesses:
            for item in weaknesses:
                print(f"⚠ {item}")
        else:
            print("None")

        print("\nRecommendation")
        print("-" * 70)

        if result.verdict == "EXCELLENT":
            print(
                "The response is highly reliable "
                "and suitable for direct use."
            )

        elif result.verdict == "GOOD":
            print(
                "The response is reliable with "
                "only minor issues."
            )

        elif result.verdict == "NEEDS_REVIEW":
            print(
                "The response should be reviewed "
                "before use because some issues "
                "were detected."
            )

        else:
            print(
                "The response is unreliable and "
                "requires significant revision."
            )

        print()
        print("=" * 70)