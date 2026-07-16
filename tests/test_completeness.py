from evaluators.relevance import RelevanceEvaluator
from evaluators.completeness import CompletenessEvaluator
from evaluators.completeness import CompletenessEvaluator


def main():

    relevance_evaluator = RelevanceEvaluator()

    evaluator = CompletenessEvaluator(
    relevance_evaluator=relevance_evaluator
)

    test_cases = [

        {
            "name": "Complete Two-Part Answer",
            "prompt": (
                "Who created Nova and when was it released?"
            ),
            "response": (
                "Nova was created by Arin and released in 2023."
            ),
        },

        {
            "name": "Creator Only",
            "prompt": (
                "Who created Nova and when was it released?"
            ),
            "response": (
                "Nova was created by Arin."
            ),
        },

        {
            "name": "Release Only",
            "prompt": (
                "Who created Nova and when was it released?"
            ),
            "response": (
                "Nova was released in 2023."
            ),
        },

        {
            "name": "Complete Three-Part",
            "prompt": (
                "Who created Nova, when was it released, "
                "and what is it used for?"
            ),
            "response": (
                "Nova was created by Arin, "
                "released in 2023, "
                "and is used for machine learning."
            ),
        },

        {
            "name": "Two Of Three",
            "prompt": (
                "Who created Nova, when was it released, "
                "and what is it used for?"
            ),
            "response": (
                "Nova was created by Arin "
                "and released in 2023."
            ),
        },

        {
            "name": "Describe And Explain",
            "prompt": (
                "Describe Nova and explain why "
                "developers use it."
            ),
            "response": (
                "Nova is a fictional programming language. "
                "Developers use it because it supports "
                "multiple paradigms."
            ),
        },

        {
            "name": "Comparison",
            "prompt": (
                "Compare Nova and Orion."
            ),
            "response": (
                "Nova supports multiple paradigms, "
                "while Orion focuses on one paradigm."
            ),
        },

    ]

    print("\nCOMPLETENESS TEST")
    print("=" * 70)

    for test in test_cases:

        result = evaluator.evaluate(
            prompt=test["prompt"],
            response=test["response"],
        )

        print(f"\n{test['name']}")
        print("-" * 70)

        print(
            "Completeness Score:",
            result["completeness_score"],
        )

        print(
            "Answered Components:",
            f"{result['answered_components']}/"
            f"{result['total_components']}"
        )


if __name__ == "__main__":
    main()