from evaluators.instruction import InstructionEvaluator


def main():

    evaluator = InstructionEvaluator()

    test_cases = [

        {
            "name": "Sentence Limit Pass",
            "prompt": "Describe Nova in no more than 2 sentences.",
            "response": (
                "Nova is a fictional programming language. "
                "It supports multiple programming paradigms."
            ),
        },

        {
            "name": "Sentence Limit Fail",
            "prompt": "Describe Nova in no more than 2 sentences.",
            "response": (
                "Nova is a programming language. "
                "It was created by Arin. "
                "It supports multiple paradigms."
            ),
        },

        {
            "name": "Word Limit Pass",
            "prompt": "Describe Nova in no more than 8 words.",
            "response": "Nova is a fictional programming language.",
        },

        {
            "name": "Word Limit Fail",
            "prompt": "Describe Nova in no more than 5 words.",
            "response": "Nova is a fictional programming language.",
        },

        {
            "name": "Bullet List Pass",
            "prompt": "Describe Nova using bullet points.",
            "response": (
                "- Fictional language\n"
                "- Supports multiple paradigms"
            ),
        },

        {
            "name": "Bullet List Fail",
            "prompt": "Describe Nova using bullet points.",
            "response": (
                "Nova is fictional. "
                "It supports multiple paradigms."
            ),
        },

        {
            "name": "JSON Pass",
            "prompt": "Return the answer in JSON.",
            "response": (
                '{"language":"Nova","creator":"Arin"}'
            ),
        },

        {
            "name": "JSON Fail",
            "prompt": "Return the answer in JSON.",
            "response": (
                "Nova was created by Arin."
            ),
        },

    ]

    print("\nINSTRUCTION TEST")
    print("=" * 70)

    for test in test_cases:

        result = evaluator.evaluate(
            prompt=test["prompt"],
            response=test["response"],
        )

        print(f"\n{test['name']}")
        print("-" * 70)

        print("Instruction Score:",
              result["instruction_score"])

        if result["violations"]:
            print("Violations:")

            for violation in result["violations"]:
                print(f"- {violation}")

        else:
            print("Violations: None")


if __name__ == "__main__":
    main()