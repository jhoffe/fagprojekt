import numpy as np
import torchaudio
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    
    def __init__(self, source, preprocessor, sort_by=None):
        """
        Dataset for loading data as defined by the given preprocessort.

        If multimodal data is used (e.g., for ASR), it is required that examples are paired such that files are located 
        in the same directory with extensions required by the preprocessor.

        Args:
            source (string): Path to the file containing the example files. Each file name should occupy its own line
                and no extension should be specified. E.g., '/path/to/data/123_456'.
            preprocessor (class): Preprocessor(s) for each modality.
            sort_by (int or None): Sort examples by the length of the modality given by sort_by in descending order.
        """

        self.source = source
        self.preprocessor = preprocessor if isinstance(preprocessor, list) else [preprocessor]
        self.sort_by = sort_by
        self.multimodal = True if isinstance(preprocessor, list) else False
        self.examples = self.load_examples(source)

    def __getitem__(self, index):
        """
        Indexes the examples attribute and yields the example as defined by the preprocessors.
        """
        path = self.examples[index]
        example = tuple(p(path) for p in self.preprocessor)
        return example, path

    def __len__(self):
        """
        Gives number of examples.
        """
        return len(self.examples)

    def load_examples(self, source):
        """
        Loads the examples file from a plain text file.

        Args:
            source (string): Path to the file containing the example files. Each file name should occupy its own line
                and no extension should be specified. E.g., '/path/to/data/123_456'.
        Returns:
            list of strings: Each element corresponds to an example file.
        """
        with open(source, 'r') as f:
            return f.read().splitlines()

    def collate(self, batch):
        """
        Passes the data from each preprocessor to the corresponding collate function.

        Args:
            batch (list): Examples in a list as returned by the __getitem__ method.
        
        Returns:
            tuple: Contains batches from each preprocessor in separate tuples if multiple objects are returned.
        """
        if isinstance(self.sort_by, int):
            sort_key = lambda x: self.preprocessor[self.sort_by].get_seq_len(x[0][self.sort_by])
            batch = sorted(batch, key=sort_key, reverse=True)
        data, filenames = zip(*batch)
        batch_data = [p.collate(b) for p, b in zip(self.preprocessor, zip(*data))]
        return batch_data, filenames
