import re
from typing import Dict, List

from evaluators.relevance import RelevanceEvaluator


class CompletenessEvaluator:
    COMPONENT_THRESHOLD = 0.75

    QUESTION_STARTERS = (
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "which",
    )

    TASK_STARTERS = (
        "describe",
        "explain",
        "list",
        "identify",
        "state",
        "give",
        "summarize",
        "define",
        "mention",
        "compare",
    )

    REASON_MARKERS = (
        "because",
        "since",
        "due to",
        "as a result of",
        "the reason",
        "reason is",
        "used for",
        "use it for",
        "use it to",
        "helps",
        "allows",
        "enables",
        "supports",
        "provides",
        "offers",
        "makes",
        "improves",
        "reduces",
        "increases",
        "simplifies",
    )

    METHOD_MARKERS = (
        "by",
        "through",
        "using",
        "via",
        "first",
        "then",
        "after",
        "before",
        "process",
        "method",
        "steps",
        "pipeline",
        "stage",
    )

    TIME_PATTERNS = (
        r"\b\d{4}\b",
        r"\b\d{3}0s\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b(?:january|february|march|april|may|june|"
        r"july|august|september|october|november|"
        r"december)\b",
        r"\b(?:monday|tuesday|wednesday|thursday|"
        r"friday|saturday|sunday)\b",
        r"\b(?:today|tomorrow|yesterday|"
        r"morning|afternoon|evening|night)\b",
        r"\b(?:last|next|this)\s+"
        r"(?:year|month|week|day|decade|century)\b",
        r"\b(?:one|two|three|four|five|six|seven|"
        r"eight|nine|ten|\d+)\s+"
        r"(?:year|years|month|months|week|weeks|"
        r"day|days|decade|decades)\s+ago\b",
        r"\b(?:early|mid|late)\s+"
        r"(?:\d{4}s|\d{1,2}(?:st|nd|rd|th)\s+century)\b",
        r"\brecently\b",
    )

    CREATOR_PATTERNS = (
        r"\bcreated by\s+(.+?)(?:[.!?]|$)",
        r"\bfounded by\s+(.+?)(?:[.!?]|$)",
        r"\bdeveloped by\s+(.+?)(?:[.!?]|$)",
        r"\binvented by\s+(.+?)(?:[.!?]|$)",
        r"\bdesigned by\s+(.+?)(?:[.!?]|$)",
        r"\bbuilt by\s+(.+?)(?:[.!?]|$)",
        r"\bauthored by\s+(.+?)(?:[.!?]|$)",
    )

    NON_LOCATION_TERMS = {
        "python",
        "java",
        "javascript",
        "c",
        "cpp",
        "c++",
        "rust",
        "go",
        "ruby",
        "swift",
        "kotlin",
        "html",
        "css",
        "sql",
        "pytorch",
        "tensorflow",
    }

    LOCATION_RELATION_PATTERNS = (
        r"\b(?:developed|created|founded|built|"
        r"located|based|established|headquartered|"
        r"born|released|launched)\s+in\s+"
        r"([A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*)*)",
        r"\b(?:located|based|headquartered)\s+at\s+"
        r"([A-Z][A-Za-z0-9.'-]*(?:\s+[A-Z][A-Za-z0-9.'-]*)*)",
        r"\b(?:near|inside|outside)\s+"
        r"([A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*)*)",
    )

    def __init__(
        self,
        relevance_evaluator: RelevanceEvaluator,
    ):
        self.relevance_evaluator = relevance_evaluator

    def evaluate(
        self,
        prompt: str,
        response: str,
    ) -> Dict:
        prompt_components = (
            self._extract_prompt_components(prompt)
        )

        component_evaluations = []

        for component in prompt_components:
            relevance_result = (
                self.relevance_evaluator.evaluate(
                    prompt=component,
                    response=response,
                )
            )

            relevance_score = relevance_result["score"]

            answer_type = self._detect_answer_type(
                component
            )

            type_validation = (
                self._validate_answer_type(
                    answer_type=answer_type,
                    response=response,
                )
            )

            relevance_passed = (
                relevance_score
                >= self.COMPONENT_THRESHOLD
            )

            answered = (
                relevance_passed
                and type_validation["passed"]
            )

            component_evaluations.append(
                {
                    "component": component,
                    "score": relevance_score,
                    "answer_type": answer_type,
                    "relevance_passed": relevance_passed,
                    "type_validation_passed": (
                        type_validation["passed"]
                    ),
                    "type_validation_reason": (
                        type_validation["reason"]
                    ),
                    "answered": answered,
                }
            )

        if not component_evaluations:
            return {
                "completeness_score": 1.0,
                "total_components": 0,
                "answered_components": 0,
                "component_evaluations": [],
            }

        answered_components = sum(
            1
            for evaluation in component_evaluations
            if evaluation["answered"]
        )

        total_components = len(
            component_evaluations
        )

        completeness_score = (
            answered_components / total_components
        )

        return {
            "completeness_score": round(
                completeness_score,
                4,
            ),
            "total_components": total_components,
            "answered_components": answered_components,
            "component_evaluations": (
                component_evaluations
            ),
        }

    @classmethod
    def _detect_answer_type(
        cls,
        component: str,
    ) -> str:
        normalized_component = (
            component.lower().strip()
        )

        if re.match(
            r"^(?:explain\s+)?why\b",
            normalized_component,
        ):
            return "REASON"

        if re.match(
            r"^(?:explain\s+)?how\b",
            normalized_component,
        ):
            return "METHOD"

        if re.match(
            r"^when\b",
            normalized_component,
        ):
            return "TIME"

        if re.match(
            r"^where\b",
            normalized_component,
        ):
            return "LOCATION"

        if re.match(
            r"^who\b",
            normalized_component,
        ):
            return "PERSON"

        if re.match(
            r"^compare\b",
            normalized_component,
        ):
            return "COMPARISON"

        return "GENERAL"

    @classmethod
    def _validate_answer_type(
        cls,
        answer_type: str,
        response: str,
    ) -> Dict:
        normalized_response = response.lower()

        if answer_type == "REASON":
            passed = any(
                marker in normalized_response
                for marker in cls.REASON_MARKERS
            )

            return {
                "passed": passed,
                "reason": (
                    "Reason evidence detected."
                    if passed
                    else
                    "No reason or explanatory "
                    "evidence detected."
                ),
            }

        if answer_type == "METHOD":
            passed = any(
                marker in normalized_response
                for marker in cls.METHOD_MARKERS
            )

            return {
                "passed": passed,
                "reason": (
                    "Method or process evidence detected."
                    if passed
                    else
                    "No method or process evidence detected."
                ),
            }

        if answer_type == "TIME":
            passed = any(
                re.search(
                    pattern,
                    normalized_response,
                    flags=re.IGNORECASE,
                )
                for pattern in cls.TIME_PATTERNS
            )

            return {
                "passed": passed,
                "reason": (
                    "Temporal evidence detected."
                    if passed
                    else
                    "No temporal evidence detected."
                ),
            }

        if answer_type == "LOCATION":
            passed = cls._contains_location_answer(
                response
            )

            return {
                "passed": passed,
                "reason": (
                    "Location evidence detected."
                    if passed
                    else
                    "No location evidence detected."
                ),
            }

        if answer_type == "PERSON":
            passed = cls._contains_creator_answer(
                response
            )

            return {
                "passed": passed,
                "reason": (
                    "Creator or responsible entity detected."
                    if passed
                    else
                    "No creator or responsible entity detected."
                ),
            }

        if answer_type == "COMPARISON":
            passed = cls._contains_comparison(
                response
            )

            return {
                "passed": passed,
                "reason": (
                    "Comparison structure detected."
                    if passed
                    else
                    "No comparison structure detected."
                ),
            }

        return {
            "passed": True,
            "reason": (
                "No specialized answer-type "
                "validation required."
            ),
        }

    @classmethod
    def _contains_creator_answer(
        cls,
        response: str,
    ) -> bool:
        for pattern in cls.CREATOR_PATTERNS:
            match = re.search(
                pattern,
                response,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            creator_text = match.group(1).strip()

            creator_text = re.sub(
                r"^(?:the|a|an)\s+",
                "",
                creator_text,
                flags=re.IGNORECASE,
            )

            if len(creator_text.split()) >= 1:
                return True

        return False

    @classmethod
    def _contains_location_answer(
        cls,
        response: str,
    ) -> bool:
        for pattern in cls.LOCATION_RELATION_PATTERNS:
            match = re.search(
                pattern,
                response,
            )

            if not match:
                continue

            location_text = (
                match.group(1).strip().lower()
            )

            if location_text in cls.NON_LOCATION_TERMS:
                continue

            return True

        return False

    @staticmethod
    def _contains_comparison(
        response: str,
    ) -> bool:
        comparison_markers = (
            "while",
            "whereas",
            "compared to",
            "compared with",
            "in contrast",
            "unlike",
            "both",
            "difference",
            "similar",
            "more than",
            "less than",
        )

        normalized_response = response.lower()

        return any(
            marker in normalized_response
            for marker in comparison_markers
        )

    @classmethod
    def _extract_prompt_components(
        cls,
        prompt: str,
    ) -> List[str]:
        cleaned_prompt = cls._clean_prompt(prompt)

        if not cleaned_prompt:
            return []

        clause_starters = (
            cls.QUESTION_STARTERS
            + cls.TASK_STARTERS
        )

        starter_pattern = "|".join(
            re.escape(starter)
            for starter in clause_starters
        )

        pattern = re.compile(
            rf"(?i)(?<!\w)"
            rf"({starter_pattern})\b"
        )

        raw_matches = list(
            pattern.finditer(cleaned_prompt)
        )

        matches = cls._filter_overlapping_starters(
            prompt=cleaned_prompt,
            matches=raw_matches,
        )

        if len(matches) <= 1:
            return [cleaned_prompt]

        components = []

        for index, match in enumerate(matches):
            start = match.start()

            if index + 1 < len(matches):
                end = matches[index + 1].start()
            else:
                end = len(cleaned_prompt)

            component = cleaned_prompt[start:end]

            component = cls._clean_component(
                component
            )

            if component:
                components.append(component)

        if len(components) <= 1:
            return [cleaned_prompt]

        subject = cls._infer_subject(
            components
        )

        restored_components = [
            cls._restore_references(
                component=component,
                subject=subject,
            )
            for component in components
        ]

        return restored_components

    @classmethod
    def _filter_overlapping_starters(
        cls,
        prompt: str,
        matches: list,
    ) -> list:
        if not matches:
            return []

        filtered_matches = []

        for match in matches:
            current_starter = (
                match.group(1).lower()
            )

            if not filtered_matches:
                filtered_matches.append(match)
                continue

            previous_match = filtered_matches[-1]

            previous_starter = (
                previous_match.group(1).lower()
            )

            text_between = prompt[
                previous_match.end():
                match.start()
            ]

            immediately_follows = (
                not text_between.strip()
            )

            is_task_question_overlap = (
                previous_starter
                in cls.TASK_STARTERS
                and current_starter
                in cls.QUESTION_STARTERS
                and immediately_follows
            )

            if is_task_question_overlap:
                continue

            filtered_matches.append(match)

        return filtered_matches

    @staticmethod
    def _clean_prompt(
        prompt: str,
    ) -> str:
        return re.sub(
            r"\s+",
            " ",
            prompt,
        ).strip()

    @staticmethod
    def _clean_component(
        component: str,
    ) -> str:
        component = component.strip()

        component = re.sub(
            r"^[,\s]+",
            "",
            component,
        )

        component = re.sub(
            r"[,\s]+$",
            "",
            component,
        )

        component = re.sub(
            r"\s+(?:and|also)\s*$",
            "",
            component,
            flags=re.IGNORECASE,
        )

        component = component.strip(
            " ,?.!"
        )

        if not component:
            return ""

        component = (
            component[0].upper()
            + component[1:]
        )

        return component + "?"

    @classmethod
    def _infer_subject(
        cls,
        components: List[str],
    ) -> str:
        reference_words = {
            "it",
            "they",
            "them",
            "this",
            "that",
        }

        ignored_words = {
            "is",
            "are",
            "was",
            "were",
            "created",
            "released",
            "used",
            "for",
            "by",
            "the",
            "a",
            "an",
            "developers",
            "advantages",
            "disadvantages",
        }

        for component in components:
            words = re.findall(
                r"\b[A-Za-z][A-Za-z0-9_-]*\b",
                component,
            )

            for word in words:
                lowercase_word = word.lower()

                if (
                    lowercase_word
                    in cls.QUESTION_STARTERS
                ):
                    continue

                if (
                    lowercase_word
                    in cls.TASK_STARTERS
                ):
                    continue

                if lowercase_word in ignored_words:
                    continue

                if lowercase_word in reference_words:
                    continue

                if word[0].isupper():
                    return word

        return ""

    @staticmethod
    def _restore_references(
        component: str,
        subject: str,
    ) -> str:
        if not subject:
            return component

        restored_component = component

        patterns = [
            (
                r"\bit\b",
                subject,
            ),
            (
                r"\bthis\b",
                subject,
            ),
            (
                r"\bthat\b",
                subject,
            ),
        ]

        for pattern, replacement in patterns:
            restored_component = re.sub(
                pattern,
                replacement,
                restored_component,
                flags=re.IGNORECASE,
            )

        return restored_component