from typing import Dict, List

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from config import EMBEDDING_MODEL_NAME
from utils.text_processing import split_sentences


class EvidenceRetriever:
    def __init__(self):
        print("Loading evidence retrieval model...")

        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        print("Evidence retrieval model loaded.")

    def retrieve(
        self,
        claim: str,
        context: str,
        top_k: int = 5,
    ) -> List[Dict]:
        context_sentences = split_sentences(context)

        if not claim.strip() or not context_sentences:
            return []

        evidence_windows = self._create_evidence_windows(
            context_sentences
        )

        claim_embedding = self.model.encode(
            claim,
            convert_to_tensor=True,
        )

        evidence_texts = [
            window["evidence"]
            for window in evidence_windows
        ]

        evidence_embeddings = self.model.encode(
            evidence_texts,
            convert_to_tensor=True,
        )

        similarity_scores = cos_sim(
            claim_embedding,
            evidence_embeddings,
        )[0]

        ranked_indices = similarity_scores.argsort(
            descending=True
        )

        results = []

        for index in ranked_indices[:top_k]:
            window_index = index.item()

            score = similarity_scores[
                window_index
            ].item()

            results.append(
                {
                    "evidence": evidence_windows[
                        window_index
                    ]["evidence"],
                    "similarity_score": round(
                        score,
                        4,
                    ),
                    "window_type": evidence_windows[
                        window_index
                    ]["window_type"],
                }
            )

        return results

    @staticmethod
    def _create_evidence_windows(
        sentences: List[str],
    ) -> List[Dict]:
        windows = []

        for index, sentence in enumerate(sentences):
            windows.append(
                {
                    "evidence": sentence,
                    "window_type": "SINGLE",
                }
            )

            if index > 0:
                contextual_evidence = (
                    f"{sentences[index - 1]} "
                    f"{sentence}"
                )

                windows.append(
                    {
                        "evidence": contextual_evidence,
                        "window_type": "CONTEXTUAL",
                    }
                )

        return windows