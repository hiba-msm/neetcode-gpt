from typing import List
from collections import Counter


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            n = len(tokens)
            if n < 2:
                break

            # Faster than manual Python counting
            counts = Counter(zip(tokens, tokens[1:]))

            # Highest frequency, lexicographically smallest tie
            best_pair = min(counts.items(), key=lambda x: (-x[1], x[0]))[0]
            a, b = best_pair
            merged = a + b
            merges.append([a, b])

            # Merge non-overlapping occurrences
            new_tokens = []
            append = new_tokens.append
            i = 0
            last = n - 1

            while i < last:
                if tokens[i] == a and tokens[i + 1] == b:
                    append(merged)
                    i += 2
                else:
                    append(tokens[i])
                    i += 1

            if i == last:
                append(tokens[i])

            tokens = new_tokens

        return merges