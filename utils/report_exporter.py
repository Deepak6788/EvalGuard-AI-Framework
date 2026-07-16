import json
from datetime import datetime
from pathlib import Path


class ReportExporter:

    @staticmethod
    def export_json(result):

        reports_folder = Path("reports")
        reports_folder.mkdir(exist_ok=True)

        now = datetime.now()

        timestamp = now.strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = (
            reports_folder
            / f"evalguard_report_{timestamp}.json"
        )

        report = {
            "generated_at": now.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "prompt": result.prompt,
            "response": result.response,
            "context": result.context,
            "verdict": result.verdict,
            "overall_score": result.overall_score,
            "relevance_score": result.relevance_score,
            "instruction_score": result.instruction_score,
            "completeness_score": result.completeness_score,
            "groundedness_score": result.groundedness_score,
            "hallucination_risk": result.hallucination_risk,
            "issues": result.issues,
            "claims": [],
        }

        for claim in result.claim_evaluations:

            report["claims"].append(
                {
                    "claim": claim.claim,
                    "evidence": claim.evidence,
                    "label": claim.nli_label,
                    "entailment_score": claim.entailment_score,
                    "neutral_score": claim.neutral_score,
                    "contradiction_score": claim.contradiction_score,
                }
            )

        with open(
            filename,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
                ensure_ascii=False,
            )

        print()
        print("=" * 70)
        print("Report Export")
        print("-" * 70)
        print(
            f"Saved Successfully : {filename}"
        )
        print(
            f"Generated At       : "
            f"{report['generated_at']}"
        )
        print("=" * 70)