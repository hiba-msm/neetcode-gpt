from typing import List, Dict


class Solution:
    def _greedy_tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0
        n = len(text)

        # Small optimization: never try longer than the longest vocab token
        max_len = max((len(tok) for tok in vocab), default=1)

        while i < n:
            best = None
            end = min(n, i + max_len)

            for j in range(end, i, -1):
                piece = text[i:j]
                if piece in vocab:
                    best = piece
                    break

            if best is None:
                best = text[i]

            tokens.append(best)
            i += len(best)

        return tokens

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        return [self._greedy_tokenize(str(num), vocab) for num in numbers]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        return len(self._greedy_tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        words = text.split()

        if not words:
            return 0.0

        return round(self.count_tokens(text, vocab) / len(words), 4)