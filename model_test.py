from typing import Iterator, Iterable, Tuple, Text, Union
from typing import List, Set, Dict, Tuple, Optional, Text
import re
import numpy as np
from scipy.sparse import spmatrix
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import html
import matplotlib.pyplot as plt
from sklearn import metrics
import random
from sklearn import svm

import time

FLAGS = re.MULTILINE | re.DOTALL
NDArray = Union[np.ndarray, spmatrix]

# traincount = 10000
# devcount = 500

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

def nltkTweetTokenizer(text):
    # must include "from nltk.tokenize import TweetTokenizer"
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text)
    return tokens

def dropStopWords(tokenList, stopWords):
    tokens_without_sw = [word for word in tokenList if not word in stopWords]
    return tokens_without_sw

def read_cleaned_file(file_path, printBool=False, max_num_posts=50000):
    list_items = []
    post_counter = 0
    with open(file_path) as inp:
        inp_reader = csv.reader(inp, delimiter=',')
        next(inp_reader) # skip header
        for line in inp_reader:
            post_counter += 1
            post_tuple = (line[1], line[0])
            list_items.append(post_tuple)
            if post_counter >= max_num_posts:
                break
    print("number of posts read:", post_counter)
    return list_items


def selectPostsRandom(postList, labelList, numSamples=50):
    selectedPosts = []
    selectedLabels = []
    maxIndex = len(postList)-1
    for i in range(numSamples):
        ind = random.randint(0,maxIndex)
        post = postList[ind]
        lab = labelList[ind]
        selectedPosts.append(post)
        selectedLabels.append(lab)

    return (selectedPosts,selectedLabels)





class TextToFeatures:
    def __init__(self, texts: Iterable[Text]):
        """Initializes an object for converting texts to features.

        During initialization, the provided training texts are analyzed to
        determine the vocabulary, i.e., all feature values that the converter
        will support. Each such feature value will be associated with a unique
        integer index that may later be accessed via the .index() method.

        It is up to the implementer exactly what features to produce from a
        text, but the features will always include some single words and some
        multi-word expressions (e.g., "need" and "to you").

        :param texts: The training texts.
        """
        self.mapping_wordkey = {}
        self.mapping_idxkey = {}
        # naive tokenization
        temp_set_words = set() # sets do not contain duplicates, so we can collect all of the words first
        last_word = ""
        for line in texts:
            # words = [word.lower() for word in line.split()]
            # words = tokenize_text(line)
            words = nltkTweetTokenizer(line)
            # print(words)
            for word in words:
                temp_set_words.add(word)
                if last_word != "":
                    temp_set_words.add(last_word + " " + word)

                last_word = word

        i = 0
        for token in temp_set_words:
            self.mapping_idxkey[i] = token
            self.mapping_wordkey[token] = i
            i += 1

        self.num_features = len(temp_set_words)



    def index(self, feature: Text):
        """Returns the index in the vocabulary of the given feature value.

        :param feature: A feature
        :return: The unique integer index associated with the feature.
        """
        if feature in self.mapping_wordkey.keys():
            return self.mapping_wordkey[feature]
        else:
            return -1


    def __call__(self, texts: Iterable[Text]) -> NDArray:
        """Creates a feature matrix from a sequence of texts.

        Each row of the matrix corresponds to one of the input texts. The value
        at index j of row i is the value in the ith text of the feature
        associated with the unique integer j.

        It is up to the implementer what the value of a feature that is present
        in a text should be, though a common choice is 1. Features that are
        absent from a text will have the value 0.

        :param texts: A sequence of texts.
        :return: A matrix, with one row of feature values for each text.
        """
        outer_list = []# rows
        for i in range(len(texts)):
            inner_list = [0 for j in range(self.num_features)] # columns
            # sentence_split = tokenize_text(texts[i])
            sentence_split = nltkTweetTokenizer(texts[i])
            feature_list = []
            last_word = ""
            for i in range(len(sentence_split)):
                if i != 0:
                    combin = last_word + " " + sentence_split[i]
                    feature_list.append(combin)
                feature_list.append(sentence_split[i])

                last_word = sentence_split[i]
            
            for k in range(self.num_features):
                if self.mapping_idxkey[k] in feature_list:
                    inner_list[k] += 1

            outer_list.append(inner_list)

        return np.array(outer_list)

class TextToLabels:
    def __init__(self, labels: Iterable[Text]):
        """Initializes an object for converting texts to labels.

        During initialization, the provided training labels are analyzed to
        determine the vocabulary, i.e., all labels that the converter will
        support. Each such label will be associated with a unique integer index
        that may later be accessed via the .index() method.

        :param labels: The training labels.
        """
        self.mapping_labkey = {}
        self.mapping_idxkey = {}
        # naive tokenization
        temp_set_labels = set() # sets do not contain duplicates, so we can collect all of the words first
        for lab in labels:
            temp_set_labels.add(lab)

        i = 0
        for lab in temp_set_labels:
            self.mapping_idxkey[i] = lab
            self.mapping_labkey[lab] = i
            i += 1

    def index(self, label: Text) -> int:
        """Returns the index in the vocabulary of the given label.

        :param label: A label
        :return: The unique integer index associated with the label.
        """
        return self.mapping_labkey[label]

    def __call__(self, labels: Iterable[Text]) -> NDArray:
        """Creates a label vector from a sequence of labels.

        Each entry in the vector corresponds to one of the input labels. The
        value at index j is the unique integer associated with the jth label.

        :param labels: A sequence of labels.
        :return: A vector, with one entry for each label.
        """
        return [self.index(l) for l in labels]


class Classifier:
    def __init__(self, typeStr):
        """
            Initalizes a logistic regression classifier.
        """
        if typeStr == "logreg":
            self.model = LogisticRegression()
        elif typeStr == "rfc":
            self.model = RandomForestClassifier(n_estimators=200,bootstrap=True)
        elif typeStr == "svm":
            weights = {"biased": 0.6, "unbiased":0.4}
            self.model = svm.SVC(cache_size=1000)

    def train(self, features: NDArray, labels: NDArray) -> None:
        """Trains the classifier using the given training examples.

        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :param labels: A label vector, where each entry represents a label.
        Such vectors will typically be generated via TextToLabels.
        """
        best_clf = self.model.fit(features, labels)

        return None

    def predict(self, features: NDArray) -> NDArray:
        """Makes predictions for each of the given examples.

        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :return: A prediction vector, where each entry represents a label.
        """
        vector = self.model.predict(features)
        # print(vector.score)
        # print(metrics.f1_score(y_test, vector, average="macro"))

        # n = np.array([0,0])
        return vector


def runPrediction(train_texts,train_labels,dev_texts,dev_labels,train_select,dev_select):
    train_texts,train_labels = selectPostsRandom(train_texts,train_labels,train_select)
    dev_texts,dev_labels = selectPostsRandom(dev_texts,dev_labels,dev_select)

    # create the feature extractor from the training texts
    text_to_features = TextToFeatures(train_texts)

    # create the label encoder from the training texts
    text_to_labels = TextToLabels(train_labels)

    # train the classifier on the training data
    # classifierType = "logreg"
    # classifierType = "rfc"
    classifierType = "svm"
    classifier = Classifier(classifierType)
    classifier.train(text_to_features(train_texts), text_to_labels(train_labels))

    # make predictions on the development data
    predicted_indices = classifier.predict(text_to_features(dev_texts))

    # measure performance of predictions
    devel_indices = text_to_labels(dev_labels)
    bias_label = text_to_labels.index("biased")
    f1 = f1_score(devel_indices, predicted_indices, pos_label=bias_label)
    accuracy = accuracy_score(devel_indices, predicted_indices)
    prec = precision_score(devel_indices, predicted_indices)
    rec = recall_score(devel_indices, predicted_indices)


    cm = metrics.confusion_matrix(devel_indices, predicted_indices)
    print(cm)

    print("F1: ", f1)
    print("Accuracy: ", accuracy)
    print("Precision: ", prec)
    print("Recall: ", rec)

    makeCMFigure(cm, classifierType,f1,prec,rec,train_select,dev_select)

    return (f1,prec,rec)

def Average(lst):
    return sum(lst) / len(lst)

def makeCMFigure(cm, modelType, f1_score, prec_score, rec_score, traincount, devcount):
    suptit = "Confusion matrix, " + modelType
    subtit = "Trained on " + str(traincount) + " samples\nTested on " + str(devcount) + " samples from test set\n"
    subtit += "F1: " + str(f1_score)[0:6] + "\n"
    subtit += "Precision: " + str(prec_score)[0:6] + ", Recall: " + str(rec_score)[0:6] + "\n"
    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.suptitle(suptit, size = 18)
    plt.title(subtit, size = 13)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["biased", "unbiased"], size = 10)
    plt.yticks(tick_marks, ["biased", "unbiased"], size = 10)
    plt.tight_layout()
    plt.ylabel('Actual label', size = 15, labelpad=15)
    plt.xlabel('Predicted label', size = 15, labelpad=15)
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
            horizontalalignment='center',
            verticalalignment='center')
    # plt.autoscale() 
    plt.tight_layout()
    plt.show()


def main():
    # SLUR_LIST = compileSlurs() # compiles five .txt files containing lists of slurs by category into one comprehensive .txt list
    # nltk.download('stopwords')
    # nltk.download('punkt')
    stopWords = stopwords.words('english')
    stopWords.remove('the') # this word seems to be critical for identifying discrimatory intent
    # stopWords.remove('not') # important for meaning
    # print(stopWords)

    # get texts and labels from the training data
    train_path = "cleaned_train.csv"
    train_list = read_cleaned_file(train_path, True)
    train_texts = [obj[1] for obj in train_list]
    train_labels = [obj[0] for obj in train_list]
    # train_texts,train_labels = selectPostsRandom(train_texts,train_labels,1000)

    # get texts and labels from the development data
    dev_path = "cleaned_dev.csv"
    dev_list = read_cleaned_file(dev_path, True)
    dev_texts = [obj[1] for obj in dev_list]
    dev_labels = [obj[0] for obj in dev_list]
    # dev_texts,dev_labels = selectPostsRandom(dev_texts,dev_labels,200)

    # get texts and labels from the development data
    tst_path = "cleaned_test.csv"
    tst_list = read_cleaned_file(tst_path, True)
    tst_texts = [obj[1] for obj in tst_list]
    tst_labels = [obj[0] for obj in tst_list]

    f1_list = []
    prec_list = []
    rec_list = []

    # print("DEVELOPMENT SET")
    # f1, precision, recall = runPrediction(train_texts,train_labels,dev_texts,dev_labels,10000,1000)
    '''
    for i in range(0,1):
        f1, precision, recall = runPrediction(train_texts,train_labels,dev_texts,dev_labels,10000,1000)
        f1_list.append(f1)
        prec_list.append(precision)
        rec_list.append(recall)

    '''

    print("\nTEST SET")
    f1, precision, recall = runPrediction(train_texts,train_labels,tst_texts,tst_labels,10000,1000)



    # print("The average f1:", Average(f1_list))
    # print("The average precision:", Average(prec_list))
    # print("The average recall:", Average(rec_list))





# below is for testing purposes
if __name__ == '__main__':
    # read_ptbtagged("PTBSmall/train.tagged")
    # read_ptbtagged("PTBSmall/dev.tagged")
    start_time = time.time()
    main()
    secondsElapsed = (time.time() - start_time)
    minutesElapsed = secondsElapsed / 60
    print("--- ran in %s seconds ---" % secondsElapsed)
    if minutesElapsed > 2:
        print("which is about %s minutes" % minutesElapsed)
    # main()