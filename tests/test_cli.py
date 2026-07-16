from utils.cli import EvalGuardCLI


def main():

    cli = EvalGuardCLI()

    cli.show_banner()

    cli.show_evaluating()

    cli.show_completion()


if __name__ == "__main__":
    main()