import csv

class StatTracker:
    def __init__(self, columns, output_file):
        self.columns = columns
        self.stats = []
        self.output_file = output_file
        self.i = 1

        with open(self.output_file, 'w') as f:
            csvwriter = csv.writer(f, dialect='unix', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(self.columns)

    def track(self, stat):
        assert len(stat) == len(self.columns)
        self.stats.append(stat)

        self.i += 1

        if self.i % 1 == 0:
            self.write()

        return self

    def write(self):
        if len(self.stats) == 0:
            return self

        with open(self.output_file, 'a') as f:
            csvwriter = csv.writer(f, dialect='unix')
            csvwriter.writerows(self.stats)
        self.flush()

        return self

    def flush(self):
        self.stats = []

        return self