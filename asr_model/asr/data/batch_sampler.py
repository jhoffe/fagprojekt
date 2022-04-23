import numpy as np

class UniformBatchSampler:
    def __init__(self, dataset_size: int, num_steps: int, batch_size: int):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.world = np.arange(self.dataset_size)

    def __iter__(self):
        for _ in range(self.__len__()):
            batch = np.random.choice(self.world, self.batch_size, replace=False).tolist()
            yield batch

    def __len__(self) -> int:
        return self.num_steps