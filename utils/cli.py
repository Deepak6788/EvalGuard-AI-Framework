class EvalGuardCLI:
    @staticmethod
    def show_banner():
        print("=" * 70)
        print("                      EVALGUARD")
        print("             AI Response Evaluation Framework")
        print("=" * 70)
        print()

    @staticmethod
    def get_user_input():
        EvalGuardCLI.show_banner()

        print("Enter the Prompt:")
        prompt = input("> ").strip()

        print("\nEnter the AI Response:")
        response = input("> ").strip()

        print("\nEnter the Reference Context")
        print("(Type END on a new line to finish)\n")

        context_lines = []

        while True:
            line = input()

            if line.strip().upper() == "END":
                break

            context_lines.append(line)

        context = "\n".join(context_lines)

        return {
            "prompt": prompt,
            "response": response,
            "context": context,
        }

    @staticmethod
    def show_evaluating():
        print()
        print("Evaluating response...")
        print("-" * 70)

    @staticmethod
    def show_completion():
        print("-" * 70)
        print("Evaluation completed successfully.")
        print("=" * 70)