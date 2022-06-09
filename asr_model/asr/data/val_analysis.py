from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ValidationAnalysis:
    def __init__(self, data=None):
        nltk.download('stopwords')
        self.data = data if data is not None else []
        self.df = None

        if data is not None:
            self._create_df()

    def append(self, path, guess, wer, cer):
        speaker_id, chapter_id, utterance_id = [id[1:] for id in path.split('/')[-1].split('_')]

        with open(path + ".txt", "r") as f:
            text = f.read()

        self.data.append([path, speaker_id, chapter_id, utterance_id, text, guess, wer, cer])

        return self

    def append_batch(self, paths, guesses, wers, cers):
        for args in zip(paths, guesses, wers, cers):
            self.append(*args)

        return self

    def _create_df(self):
        self.df = pd.DataFrame(data=self.data,
                               columns=["Path", "SpeakerId", "ChapterId", "UtteranceId", "Text", "Guess", "WER", "CER"])

        return self

    def preprocess(self):
        self._create_df()

        self._count_words()
        self._count_stopwords_for_all()

        return self

    def _count_words(self):
        self.df["WordCount"] = self.df["Text"].apply(lambda t: len(t.split(" ")))

        return self

    @staticmethod
    def _count_stopwords(text):
        words = text.split(" ")

        stopword_count = 0
        sw = stopwords.words('english')

        for word in words:
            if word.lower() in sw:
                stopword_count += 1

        return stopword_count

    def _count_stopwords_for_all(self):
        self.df["StopWordCount"] = self.df["Text"].apply(self._count_stopwords)

        return self

    def plot(self):
        #WER/Stopword
        wer_stopword = plt.figure()
        plt.scatter(self.df["StopWordCount"], self.df["WER"])
        plt.suptitle("WER and stop word count")

        plt.xlabel("stop words")
        plt.ylabel("WER")

        # WER/wordcounts
        wer_wordcounts = plt.figure()
        plt.scatter(self.df["WordCount"], self.df["WER"])
        plt.suptitle("WER and word count")

        plt.xlabel("word count")
        plt.ylabel("WER")

        # Stopword counts histograms
        stopword_hist = plt.figure()
        sns.histplot(data=self.df["StopWordCount"])

        # word counts histograms
        wordcount_hist = plt.figure()
        sns.histplot(data=self.df["WordCount"])

        # WER histograms
        wer_hist = plt.figure()
        sns.histplot(data=self.df["WER"])

        # CER histograms
        cer_hist = plt.figure()
        sns.histplot(data=self.df["CER"])

        return wer_stopword, wer_wordcounts, stopword_hist, wordcount_hist, wer_hist, cer_hist