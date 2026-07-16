import re
from typing import Dict

import torch
from sentence_transformers import CrossEncoder

from config import RELEVANCE_MODEL_NAME


class RelevanceEvaluator:
    def __init__(self):
        print("Loading relevance cross-encoder...")

        self.model = CrossEncoder(
            RELEVANCE_MODEL_NAME,
            activation_fn=torch.nn.Sigmoid(),
        )

        print("Relevance cross-encoder loaded.")

    def evaluate(
        self,
        prompt: str,
        response: str,
    ) -> Dict:
        if not prompt.strip() or not response.strip():
            return {
                "score": 0.0,
                "label": "LOW",
                "semantic_prompt": "",
            }

        semantic_prompt = self._extract_semantic_prompt(
            prompt
        )

        prediction = self.model.predict(
            [
                (
                    semantic_prompt,
                    response,
                )
            ]
        )

        relevance_score = float(prediction[0])

        normalized_score = max(
            0.0,
            min(
                1.0,
                relevance_score,
            ),
        )

        label = self._get_label(
            normalized_score
        )

        return {
            "score": round(
                normalized_score,
                4,
            ),
            "label": label,
            "semantic_prompt": semantic_prompt,
        }

    @staticmethod
    def _extract_semantic_prompt(
        prompt: str,
    ) -> str:
        semantic_prompt = prompt.strip()

        patterns = [
            (
                r"\busing\s+no\s+more\s+than\s+"
                r"\d+\s+sentences?\b",
                "",
            ),
            (
                r"\bin\s+no\s+more\s+than\s+"
                r"\d+\s+sentences?\b",
                "",
            ),
            (
                r"\bwithin\s+\d+\s+sentences?\b",
                "",
            ),
            (
                r"\busing\s+\d+\s+sentences?\b",
                "",
            ),
            (
                r"\bin\s+\d+\s+sentences?\b",
                "",
            ),
            (
                r"\bmaximum\s+\d+\s+sentences?\b",
                "",
            ),
            (
                r"\bmax\s+\d+\s+sentences?\b",
                "",
            ),
            (
                r"\busing\s+no\s+more\s+than\s+"
                r"\d+\s+words?\b",
                "",
            ),
            (
                r"\bin\s+no\s+more\s+than\s+"
                r"\d+\s+words?\b",
                "",
            ),
            (
                r"\bwithin\s+\d+\s+words?\b",
                "",
            ),
            (
                r"\bunder\s+\d+\s+words?\b",
                "",
            ),
            (
                r"\busing\s+\d+\s+words?\b",
                "",
            ),
            (
                r"\bin\s+\d+\s+words?\b",
                "",
            ),
            (
                r"\bmaximum\s+\d+\s+words?\b",
                "",
            ),
            (
                r"\bmax\s+\d+\s+words?\b",
                "",
            ),
            (
                r"\busing\s+bullet\s+points?\b",
                "",
            ),
            (
                r"\bin\s+bullet\s+points?\b",
                "",
            ),
            (
                r"\busing\s+a\s+numbered\s+list\b",
                "",
            ),
            (
                r"\bin\s+a\s+numbered\s+list\b",
                "",
            ),
            (
                r"\bin\s+(?:valid\s+)?"
                r"json(?:\s+format)?\b",
                "",
            ),
            (
                r"\bas\s+(?:valid\s+)?json\b",
                "",
            ),
        ]

        for pattern, replacement in patterns:
            semantic_prompt = re.sub(
                pattern,
                replacement,
                semantic_prompt,
                flags=re.IGNORECASE,
            )

        semantic_prompt = re.sub(
            r"\s+",
            " ",
            semantic_prompt,
        ).strip()

        semantic_prompt = re.sub(
            r"\s+([?.!,])",
            r"\1",
            semantic_prompt,
        )

        semantic_prompt = re.sub(
            r"\b(?:using|in|within)\s*([?.!]?)$",
            r"\1",
            semantic_prompt,
            flags=re.IGNORECASE,
        ).strip()

        return semantic_prompt

    @staticmethod
    def _get_label(
        score: float,
    ) -> str:
        if score >= 0.75:
            return "HIGH"

        if score >= 0.50:
            return "MEDIUM"

        return "LOW"