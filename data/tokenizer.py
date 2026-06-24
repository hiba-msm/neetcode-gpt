from typing import List
from collections import Counter


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            if len(tokens) < 2:
                break

            # Count adjacent pairs
            pair_counts = Counter()
            for i in range(len(tokens) - 1):
                pair_counts[(tokens[i], tokens[i + 1])] += 1

            if not pair_counts:
                break

            # Most frequent pair, tie broken lexicographically
            best_pair = min(pair_counts.keys(), key=lambda p: (-pair_counts[p], p))
            merges.append([best_pair[0], best_pair[1]])

            # Merge all non-overlapping occurrences left to right
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return merges