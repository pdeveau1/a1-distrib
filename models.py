# models.py

from sentiment_data import *
from utils import *

from collections import Counter
import numpy as np
import random
import string

random.seed(42)

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        # Create a counter to hold feature counts
        counter = Counter()
        # Iterate over each word in the sentence
        for word in sentence:
            # Convert word to lowercase for case insensitivity
            word = word.lower()
            # Remove punctuation from the word
            word = word.translate(str.maketrans('', '', string.punctuation))
            # Skip punctiation and RRB/LRB tokens
            if word != '' and word != 'rrb' and word != 'lrb':
                # Get the index of the word, adding it to the indexer if specified
                idx = self.indexer.add_and_get_index(word, add=add_to_indexer)
                # Only add to counter if the word is in the indexer
                if idx != -1:
                    counter[idx] += 1
        return counter

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        # Create a counter to hold feature counts
        counter = Counter()
        # Iterate over each word in the sentence
        for i in range(len(sentence)-1):
            bigram = sentence[i].lower() + " " + sentence[i+1].lower()
            # Get the index of the word, adding it to the indexer if specified
            idx = self.indexer.add_and_get_index(bigram, add=add_to_indexer)
            # Only add to counter if the word is in the indexer
            if idx != -1:
                counter[idx] += 1
        return counter


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feature_extractor: FeatureExtractor, indexer: Indexer):
        self.feature_extractor = feature_extractor
        self.indexer = indexer
        self.weights = np.zeros(len(indexer))

    def predict(self, sentence: List[str]) -> int:
        features = self.feature_extractor.extract_features(sentence, add_to_indexer=False)
        # Dot product of weights and features
        score = 0.0
        for idx, count in features.items():
            if idx < len(self.weights):
                score += self.weights[idx] * count
        return 1 if score >= 0 else 0
    
    def update(self, sentence: List[str], label: int, learning_rate: float=1.0):
        features = self.feature_extractor.extract_features(sentence, add_to_indexer=True)
        prediction = self.predict(sentence)
        error = label - prediction
        for idx, count in features.items():
            if idx >= len(self.weights):
                self.weights = np.append(self.weights, np.zeros(idx - len(self.weights) + 1))
            self.weights[idx] += learning_rate * error * count


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feature_extractor: FeatureExtractor, indexer: Indexer):
        self.feature_extractor = feature_extractor
        self.indexer = indexer
        self.weights = np.zeros(len(indexer))

    def sigmoid(self, z: float) -> float:
        return 1 / (1 + np.exp(-z))
    
    def predict_proba(self, sentence: List[str]) -> float:
        features = self.feature_extractor.extract_features(sentence, add_to_indexer=False)
        # Dot product of weights and features
        score = 0.0
        for idx, count in features.items():
            if idx < len(self.weights):
                score += self.weights[idx] * count
        return self.sigmoid(score)
    
    def predict(self, sentence: List[str]) -> int:
        prob = self.predict_proba(sentence)
        return 1 if prob >= 0.5 else 0
    
    def update(self, sentence: List[str], label: int, learning_rate: float=0.1):
        features = self.feature_extractor.extract_features(sentence, add_to_indexer=True)
        prediction = self.predict(sentence)
        error = label - prediction
        for idx, count in features.items():
            if idx >= len(self.weights):
                self.weights = np.append(self.weights, np.zeros(idx - len(self.weights) + 1))
            self.weights[idx] += learning_rate * (error * count - 0.001 * self.weights[idx])  # L2 regularization

def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, num_epochs: int=5) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    print("Training with perceptron")
    # Build the indexer
    for sentence in train_exs:
        feat_extractor.extract_features(sentence.words, add_to_indexer=True)
    
    # Initialize the perceptron classifier
    indexer = feat_extractor.get_indexer()
    model = PerceptronClassifier(feat_extractor, indexer)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("Shuffling training data")
        random.shuffle(train_exs)
        for ex in train_exs:
            model.update(ex.words, ex.label)
    return model


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, num_epochs: int=20, learning_rate: float=0.05) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    print("Training with logistic regression")

    # Build the indexer
    for sentence in train_exs:
        feat_extractor.extract_features(sentence.words, add_to_indexer=True)

    # Initialize the logistic regression classifier
    indexer = feat_extractor.get_indexer()
    model = LogisticRegressionClassifier(feat_extractor, indexer)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("Shuffling training data")
        random.shuffle(train_exs)
        for ex in train_exs:
            model.update(ex.words, ex.label, learning_rate=learning_rate)
    return model


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model