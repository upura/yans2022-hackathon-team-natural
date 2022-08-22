class DownSampler(object):
    def __init__(self, random_state):
        self.random_state = random_state

    @staticmethod
    def transform(self, data, target):
        positive_data = data[data[target] == 1]
        positive_ratio = len(positive_data) / len(data)
        negative_data = data[data[target] == 0].sample(
            frac=positive_ratio / (1 - positive_ratio), random_state=self.random_state)
        return positive_data.index.union(negative_data.index).sort_values()
