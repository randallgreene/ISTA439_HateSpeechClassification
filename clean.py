from typing import Iterator, Iterable, Tuple, Text, Union
from typing import List, Set, Dict, Tuple, Optional, Text
import re
import numpy as np
from scipy.sparse import spmatrix
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import html

FLAGS = re.MULTILINE | re.DOTALL
NDArray = Union[np.ndarray, spmatrix]

# en_model = spacy.load("en_core_web_sm")


def compileSlurs():
    files = ["slurs/disability_slurs.txt", "slurs/ethnic_slurs.txt", "slurs/gender_slurs.txt", "slurs/lgbtq_slurs.txt", "slurs/religious_slurs.txt"]
    list_slurs = []
    for slur_file in files:
        with open(slur_file, 'r') as inp:
            lines = inp.readlines()
            for line in lines:
                # line_str = line.strip()
                line_str = line.lower()
                if line_str != "\n":
                    list_slurs.append(line_str)

        print("Finished adding contents of: ", slur_file)

    set_slurs = set(list_slurs) # remove duplicates
    list_slurs = list(set_slurs)
    list_slurs.sort() # sorts alphabetically

    with open("all_slurs.txt", 'x') as f:
        f.writelines(list_slurs)

    return list_slurs

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    result = u"# {}".format(hashtag_body)
    return result

# https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

# call before tokenizing!
def removeUnwantedText(text):
    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = removeHTML(text)
    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " ") # remove URLs
    text = re_sub(r"@\w+", " usermention ")   # remove user mentions, replaced with "usermention"
    text = re_sub(r" [-+]?[.\d]*[\d]+[:,.\d]*", " number ") # removes numbers, replaced with "number"
    text = re_sub(r"\n", "") # remove new lines
    text = re_sub(r"\t", "") # remove tabs
    text = re_sub(r"\"", "") # remove quotes
    text = re_sub(r" \'", "") # remove ' that are not apostrophes
    text = re_sub(r"\' ", "") # remove ' that are not apostrophes
    text = re_sub(u"\u200f", "") # remove weird \u200f from text?
    text = remove_emojis(text)
    return text.lower()

def removeHTML(text):
    # this removes any HTML encodings like HTML emojis from text
    # must include "import html"
    return html.unescape(text)

def nltkTweetTokenizer(text):
    # must include "from nltk.tokenize import TweetTokenizer"
    tokenizer = TweetTokenizer()
    cleaned = removeUnwantedText(text)
    tokens = tokenizer.tokenize(cleaned)
    return tokens

def dropStopWords(tokenList, stopWords):
    tokens_without_sw = [word for word in tokenList if not word in stopWords]
    return tokens_without_sw

def read_corpus_file(train_path, printBool=False, max_num_posts=50000):
    list_items = []
    with open(train_path) as inp:
        inp_reader = csv.reader(inp, delimiter=',')
        post_counter = 0
        biased_count = 0
        unbiased_count = 0
        next(inp_reader) # skip header

        lab_options = ["biased", "unbiased"]
        for line in inp_reader:
            post_counter += 1
            post_tuple = (int(line[0]), lab_options[int(line[10])], line[1])
            list_items.append(post_tuple)

            if (int(line[10]) == 0):
                biased_count += 1
            if (int(line[10]) == 1):
                unbiased_count += 1

            if post_counter >= max_num_posts:
                break

    if printBool == True:
        print("number of posts = ", post_counter)
        print("checking num: ", len(list_items))
        print("biased count = ", biased_count)
        print("unbiased count = ", unbiased_count)

    return list_items


def cleanPostList(postList):
    cleanedList = []
    for post in postList:
        cleaned = removeUnwantedText(post)
        cleanedList.append(cleaned)
    return cleanedList


def writeListToCSV(filename, cleanedPosts, labels):
    data = []
    for i in range(len(cleanedPosts)):
        data.append([cleanedPosts[i],labels[i]])

    header = ["CleanedPost", "Bias/Unbiased"]

    with open(filename, 'w',  encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

    print("Wrote to csv:", filename)



def main():
    # SLUR_LIST = compileSlurs() # compiles five .txt files containing lists of slurs by category into one comprehensive .txt list
    # nltk.download('stopwords')
    # nltk.download('punkt')
    stopWords = stopwords.words('english')
    stopWords.remove('the') # this word seems to be critical for identifying discrimatory intent
    # stopWords.remove('not') # important for meaning
    # print(stopWords)

    # get texts and labels from the training data
    train_path = "SBIC.v2.agg.trn.csv"
    train_list = read_corpus_file(train_path, True)
    train_texts = [obj[2] for obj in train_list]
    train_labels = [obj[1] for obj in train_list]
    cleanedTrainTexts = cleanPostList(train_texts)

    # get texts and labels from the development data
    dev_path = "SBIC.v2.agg.dev.csv"
    dev_list = read_corpus_file(dev_path, True)
    dev_texts = [obj[2] for obj in dev_list]
    dev_labels = [obj[1] for obj in dev_list]
    cleanedDevTexts = cleanPostList(dev_texts)

    # get texts and labels from the development data
    test_path = "SBIC.v2.agg.tst.csv"
    test_list = read_corpus_file(test_path, True)
    test_texts = [obj[2] for obj in test_list]
    test_labels = [obj[1] for obj in test_list]
    cleanedTestTexts = cleanPostList(test_texts)

    writeListToCSV("cleaned_train.csv",cleanedTrainTexts,train_labels)
    writeListToCSV("cleaned_dev.csv",cleanedDevTexts,dev_labels)
    writeListToCSV("cleaned_test.csv",cleanedTestTexts,test_labels)


# below is for testing purposes
if __name__ == '__main__':
    # read_ptbtagged("PTBSmall/train.tagged")
    # read_ptbtagged("PTBSmall/dev.tagged")
    main()