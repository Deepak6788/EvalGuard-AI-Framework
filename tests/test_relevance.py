import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from evaluators.relevance import RelevanceEvaluator


def main():

    evaluator = RelevanceEvaluator()

    test_cases = [

        {
            "name": "Relevant Answer",
            "prompt": "Who created Nova?",
            "response": "Nova was created by Arin.",
        },

        {
            "name": "Partially Relevant",
            "prompt": "When was Nova released?",
            "response": "Nova supports multiple paradigms.",
        },

        {
            "name": "Irrelevant",
            "prompt": "Describe Nova.",
            "response": "The Eiffel Tower is located in Paris.",
        },

    ]

    print("\nRELEVANCE TEST")
    print("=" * 70)

    for test in test_cases:

        result = evaluator.evaluate(
            prompt=test["prompt"],
            response=test["response"],
        )

        print(f"\n{test['name']}")
        print("-" * 70)

        print("Prompt:")
        print(test["prompt"])

        print("\nResponse:")
        print(test["response"])

        print("\nScore:", result["score"])
        print("Label:", result["label"])


if __name__ == "__main__":
    main()