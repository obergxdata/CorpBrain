from collections import deque


class History:
    def __init__(self, window=8):
        self.window = window
        self.data = {}  # key -> deque

    def record(self, **kv):
        for k, v in kv.items():
            self.data.setdefault(k, deque(maxlen=self.window)).append(v)

    def last(self, key, default=0):
        d = self.data.get(key)
        return d[-1] if d else default

    def tail(self, key, n):
        d = self.data.get(key)
        if not d:
            return []
        return list(d)[-n:]

    def delta(self, key):
        d = self.data.get(key)
        if not d or len(d) < 2:
            return 0
        return d[-1] - d[-2]  # current - previous

    def trend(self, key, n=2):

        if n < 2:
            raise ValueError(f"Need at least 2 values to calculate trend, got {n}")

        values = self.tail(key, n)
        if len(values) < n:
            raise ValueError(f"{n} values are not available. Current: {len(values)}")

        # Calculate percentage changes between consecutive values
        percentage_changes = []
        for i in range(len(values) - 1):
            current = values[i]
            next_val = values[i + 1]

            # Handle division by zero
            if current == 0:
                if next_val == 0:
                    percentage_changes.append(0.0)
                else:
                    percentage_changes.append(float("inf"))
            else:
                percentage_changes.append((next_val - current) / abs(current))

        # Return average percentage change
        if not percentage_changes:
            return 0.0

        return sum(percentage_changes) / len(percentage_changes)
