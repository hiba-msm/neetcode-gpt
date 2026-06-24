from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            n = len(tokens)
            if n < 2:
                break

            # Count adjacent pairs manually: faster than Counter here
            counts = {}
            prev = tokens[0]

            for i in range(1, n):
                curr = tokens[i]
                pair = (prev, curr)
                counts[pair] = counts.get(pair, 0) + 1
                prev = curr

            # Find max frequency, lexicographically smallest tie
            best_pair = None
            best_count = -1

            for pair, count in counts.items():
                if count > best_count or (count == best_count and pair < best_pair):
                    best_pair = pair
                    best_count = count

            a, b = best_pair
            merges.append([a, b])

            # Merge non-overlapping occurrences left to right
            new_tokens = []
            i = 0

            while i < n:
                if i + 1 < n and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(a + b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return merges