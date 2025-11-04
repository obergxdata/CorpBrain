from collections import deque
import matplotlib.pyplot as plt
from pathlib import Path


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

    def chart(self, keys: list[str], name: str, log_scale: bool = False):
        """
        Plot the values for the specified keys using matplotlib.
        X-axis starts from 1 and increments for each data point.
        Saves the chart to the 'charts' directory.

        Args:
            keys: List of keys to plot from self.data
            name: Name for the chart file (without extension)
            log_scale: If True, use logarithmic scale for Y-axis
        """
        if not keys:
            print("No keys provided for charting")
            return

        # Create charts directory if it doesn't exist
        charts_dir = Path("charts")
        charts_dir.mkdir(exist_ok=True)

        # Create the plot
        plt.figure(figsize=(10, 6))

        for key in keys:
            if key not in self.data:
                print(f"Warning: Key '{key}' not found in data")
                continue

            values = list(self.data[key])
            if not values:
                print(f"Warning: No data for key '{key}'")
                continue

            # X-axis starts from 1
            x_values = list(range(1, len(values) + 1))
            plt.plot(x_values, values, label=key)

        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("History Data")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if log_scale:
            plt.yscale("log")

        plt.tight_layout()

        # Save with provided name
        filename = charts_dir / f"{name}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Chart saved to {filename}")
