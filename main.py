import time

from core.evaluation_engine import EvaluationEngine
from utils.cli import EvalGuardCLI
from utils.report_formatter import ReportFormatter
from utils.report_exporter import ReportExporter


def main():
    cli = EvalGuardCLI()

    user_input = cli.get_user_input()

    cli.show_evaluating()

    start_time = time.perf_counter()

    engine = EvaluationEngine()

    result = engine.evaluate(
        prompt=user_input["prompt"],
        response=user_input["response"],
        context=user_input["context"],
    )

    execution_time = (
        time.perf_counter() - start_time
    )

    ReportFormatter.print_report(
        result=result,
        execution_time=execution_time,
    )

    ReportExporter.export_json(result)

    cli.show_completion()


if __name__ == "__main__":
    main()