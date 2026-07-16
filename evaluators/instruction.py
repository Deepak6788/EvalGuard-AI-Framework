import json
import re
from typing import Dict, List

from utils.text_processing import split_sentences


class InstructionEvaluator:
    def evaluate(
        self,
        prompt: str,
        response: str,
    ) -> Dict:
        checks = []

        checks.extend(
            self._check_sentence_limit(
                prompt=prompt,
                response=response,
            )
        )

        checks.extend(
            self._check_word_limit(
                prompt=prompt,
                response=response,
            )
        )

        checks.extend(
            self._check_list_requirement(
                prompt=prompt,
                response=response,
            )
        )

        checks.extend(
            self._check_json_requirement(
                prompt=prompt,
                response=response,
            )
        )

        if not checks:
            return {
                "instruction_score": 1.0,
                "checks": [],
                "violations": [],
            }

        passed_checks = sum(
            1
            for check in checks
            if check["passed"]
        )

        instruction_score = (
            passed_checks / len(checks)
        )

        violations = [
            check["message"]
            for check in checks
            if not check["passed"]
        ]

        return {
            "instruction_score": round(
                instruction_score,
                4,
            ),
            "checks": checks,
            "violations": violations,
        }

    def _check_sentence_limit(
        self,
        prompt: str,
        response: str,
    ) -> List[Dict]:
        checks = []

        match = re.search(
            r"\b(?:in|using|within|maximum|max|no more than)"
            r"\s+(\d+)\s+sentences?\b",
            prompt,
            re.IGNORECASE,
        )

        if not match:
            return checks

        sentence_limit = int(match.group(1))

        sentence_count = len(
            split_sentences(response)
        )

        passed = sentence_count <= sentence_limit

        checks.append(
            {
                "instruction": "sentence_limit",
                "expected": sentence_limit,
                "actual": sentence_count,
                "passed": passed,
                "message": (
                    f"Response contains {sentence_count} "
                    f"sentences; maximum allowed is "
                    f"{sentence_limit}."
                ),
            }
        )

        return checks

    def _check_word_limit(
        self,
        prompt: str,
        response: str,
    ) -> List[Dict]:
        checks = []

        match = re.search(
            r"\b(?:in|using|within|maximum|max|under|"
            r"no more than)\s+(\d+)\s+words?\b",
            prompt,
            re.IGNORECASE,
        )

        if not match:
            return checks

        word_limit = int(match.group(1))

        word_count = len(
            re.findall(
                r"\b\w+\b",
                response,
            )
        )

        passed = word_count <= word_limit

        checks.append(
            {
                "instruction": "word_limit",
                "expected": word_limit,
                "actual": word_count,
                "passed": passed,
                "message": (
                    f"Response contains {word_count} "
                    f"words; maximum allowed is "
                    f"{word_limit}."
                ),
            }
        )

        return checks

    def _check_list_requirement(
        self,
        prompt: str,
        response: str,
    ) -> List[Dict]:
        checks = []

        list_requested = re.search(
            r"\b(?:bullet|bullets|bullet points|"
            r"numbered list|list format)\b",
            prompt,
            re.IGNORECASE,
        )

        if not list_requested:
            return checks

        lines = response.splitlines()

        list_pattern = re.compile(
            r"^\s*(?:[-*•]|\d+[.)])\s+"
        )

        has_list = any(
            list_pattern.match(line)
            for line in lines
        )

        checks.append(
            {
                "instruction": "list_format",
                "expected": True,
                "actual": has_list,
                "passed": has_list,
                "message": (
                    "Response does not use the "
                    "requested list format."
                ),
            }
        )

        return checks

    def _check_json_requirement(
        self,
        prompt: str,
        response: str,
    ) -> List[Dict]:
        checks = []

        json_requested = re.search(
            r"\bjson\b",
            prompt,
            re.IGNORECASE,
        )

        if not json_requested:
            return checks

        is_valid_json = True

        try:
            json.loads(response.strip())
        except (
            json.JSONDecodeError,
            TypeError,
        ):
            is_valid_json = False

        checks.append(
            {
                "instruction": "json_format",
                "expected": True,
                "actual": is_valid_json,
                "passed": is_valid_json,
                "message": (
                    "Response is not valid JSON."
                ),
            }
        )

        return checks