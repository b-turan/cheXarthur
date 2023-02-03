import torch


class MetricsCollector:
    def __init__(self, *metrics) -> None:
        assert all(
            isinstance(metric, str) for metric in metrics
        ), f"Expected all metrics to be of type `str`, got {type(metrics)}"
        # Tuple of strings
        self.metrics = metrics
        self.metric_dict = {}
        # Initializes the averager for each passed metric
        for metric_name in self.metrics:
            self.metric_dict[metric_name] = Averager()

    @torch.no_grad()
    def add(self, **kwargs) -> None:
        """Adds values to Averager for each metric corresponding to the keywords"""
        for metric_name, val in kwargs.items():
            self.metric_dict[metric_name].add(val)

    def print_results(self, prefix) -> None:
        """Prints the current average of each metric"""
        print(10 * "---")
        for metric_name in self.metrics:
            print(f"{prefix} {metric_name}: {self.metric_dict[metric_name].result():1.2f}")

    @torch.no_grad()
    def average(self, metric: str) -> float:
        """Returns current average of metric"""
        assert (
            metric in self.metric_dict.keys()
        ), f"Passed argument is not a key of the corresponding dictionary, got `{metric}`"
        return self.metric_dict[metric].result()

    def __repr__(self):
        return f"The metric that are collected are {self.metrics}"


class Averager:
    def __init__(self):
        """Averager class for calculating the average of a given value.
        Example:
            >>> avg = Averager()
            >>> avg.add(1)
            >>> avg.add(2)
            >>> avg.result()
            1.5
        """
        self.reset()

    def reset(self):
        """Resets the averager."""
        self.sum = 0
        self.count = 0

    def result(self):
        """Returns the average of the added values."""
        assert (
            self.count >= 1
        ), f"You have not added a value to the averager so far, Counter = {self.count}"

        return self.sum / self.count

    def add(self, val, n=1):
        """Adds a value to the averager."""
        self.sum += val
        self.count += n
