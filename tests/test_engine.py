from core.evaluation_engine import EvaluationEngine


def main():

    engine = EvaluationEngine()

    prompt = (
        "Who created Nova and when was it released?"
    )

    response = (
        "Nova was created by Arin and released in 2023."
    )

    context = (
        "Nova was created by Arin. "
        "The programming language was released in 2023."
    )

    result = engine.evaluate(
        prompt=prompt,
        response=response,
        context=context,
    )

    print("\nENGINE TEST")
    print("=" * 70)

    print(f"Verdict: {result.verdict}")
    print(f"Overall Score: {result.overall_score:.4f}")
    print(f"Relevance: {result.relevance_score:.4f}")
    print(f"Instruction: {result.instruction_score:.4f}")
    print(f"Completeness: {result.completeness_score:.4f}")
    print(f"Groundedness: {result.groundedness_score:.4f}")
    print(f"Hallucination Risk: {result.hallucination_risk:.4f}")

    print("\nIssues:")

    if result.issues:
        for issue in result.issues:
            print(f"- {issue}")
    else:
        print("None")


if __name__ == "__main__":
    main()