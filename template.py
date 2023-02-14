"""
Foundations of Natural Language Processing

Assignment 1

Please complete functions, based on their doc_string description
and instructions of the assignment. 

To test your code run:

  [hostname]s1234567 python3 s1234567.py
or
  [hostname]s1234567 python3 -i s1234567.py

The latter is useful for debugging, as it will allow you to access many
 useful global variables from the python command line

*Important*: Before submission be sure your code works _on a DICE machine_
with the --answers flag:

  [hostname]s1234567 python3 s1234567.py --answers

Also use this to generate the answers.py file you need for the interim
checker.

Best of Luck!
"""
from collections import defaultdict, Counter
from typing import Tuple, List, Any, Set, Dict, Callable

import numpy as np  # for np.mean() and np.std()
import nltk, sys, inspect
import nltk.corpus.util
from nltk import MaxentClassifier
from nltk.corpus import brown, ppattach  # import corpora

# Import LgramModel
from nltk_model import *

# Import the Twitter corpus
from twitter.twitter import *

twitter_file_ids = "20100128.txt"
assert twitter_file_ids in xtwc.fileids()

# Some helper functions

import matplotlib.pyplot as plt

def hist(hh: List[float], title: str, align: str = 'mid',
         log: bool = False, block: bool = False):
  """
  Show a histgram with bars showing mean and standard deviations
  :param hh: the data to plot
  :param title: the plot title
  :param align: passed to pyplot.hist, q.v.
  :param log: passed to pyplot.hist, q.v.  If present will be added to title
  """
  hax=plt.subplots()[1] # Thanks to https://stackoverflow.com/a/7769497
  sdax=hax.twiny()
  hax.hist(hh,bins=30,color='lightblue',align=align,log=log)
  hax.set_title(title+(' (log plot)' if log else ''))
  ylim=hax.get_ylim()
  xlim=hax.get_xlim()
  m=np.mean(hh)
  sd=np.std(hh)
  sdd=[(i,m+(i*sd)) for i in range(int(xlim[0]-(m+1)),int(xlim[1]-(m-3)))]
  for s,v in sdd:
       sdax.plot([v,v],[0,ylim[0]+ylim[1]],'r' if v==m else 'pink')
  sdax.set_xlim(xlim)
  sdax.set_ylim(ylim)
  sdax.set_xticks([v for s,v in sdd])
  sdax.set_xticklabels([str(s) for s,v in sdd])
  plt.show(block=block)


def compute_accuracy(classifier, data: List[Tuple[List, str]]) -> float:
    """
    Computes accuracy (range 0 - 1) of a classifier.
    :type classifier: e.g. NltkClassifierWrapper or NaiveBayes
    :param classifier: the classifier whose accuracy we compute.
    :param data: A list with tuples of the form (list with features, label)
    :return accuracy (range 0 - 1).
    """
    correct = 0
    for d, gold in data:
        predicted = classifier.classify(d)
        correct += predicted == gold
    return correct/len(data)


def apply_extractor(extractor_f: Callable[[str, str, str, str, str], List[Any]], data: List[Tuple[Tuple[str], str]])\
        -> List[Tuple[List[Any], str]]:
    """
    Helper function:
    Apply a feature extraction method to a labeled dataset.
    :param extractor_f: the feature extractor, that takes as input V, N1, P, N2 (all strings) and returns a list of features
    :param data: a list with tuples of the form (id, V, N1, P, N2, label)

    :return a list with tuples of the form (list with features, label)
    """
    r = []
    for d in data:
        r.append((extractor_f(*d[1:-1]), d[-1]))
    return r


def get_annotated_tweets():
    """
    :rtype list(tuple(list(str), bool))
    :return: a list of tuples (tweet, a) where tweet is a tweet preprocessed by us,
    and a is True, if the tweet is in English, and False otherwise.
    """
    import ast
    with open("twitter/annotated_dev_tweets.txt") as f:
        return [ast.literal_eval(line) for line in f.readlines()]


class NltkClassifierWrapper:
    """
    This is a little wrapper around the nltk classifiers so that we can interact with them
    in the same way as the Naive Bayes classifier.
    """
    def __init__(self, classifier_class: nltk.classify.api.ClassifierI, train_features: List[Tuple[List[Any], str]], **kwargs):
        """

        :param classifier_class: the kind of classifier we want to create an instance of.
        :param train_features: A list with tuples of the form (list with features, label)
        :param kwargs: additional keyword arguments for the classifier, e.g. number of training iterations.
        :return None
        """
        self.classifier_obj = classifier_class.train(
            [(NltkClassifierWrapper.list_to_freq_dict(d), c) for d, c in train_features], **kwargs)

    @staticmethod
    def list_to_freq_dict(d: List[Any]) -> Dict[Any, int]:
        """
        :param d: list of features

        :return: dictionary with feature counts.
        """
        return Counter(d)

    def classify(self, d: List[Any]) -> str:
        """
        :param d: list of features

        :return: most likely class
        """
        return self.classifier_obj.classify(NltkClassifierWrapper.list_to_freq_dict(d))

    def show_most_informative_features(self, n = 10):
        self.classifier_obj.show_most_informative_features(n)

# End helper functions

# ==============================================
# Section I: Language Identification [60 marks]
# ==============================================

# Question 1.1 [7.5 marks]
def train_LM(corpus: nltk.corpus.CorpusReader) -> LgramModel:
    """
    Build a bigram letter language model using LgramModel
    based on the lower-cased all-alpha subset of the entire corpus

    :param corpus: An NLTK corpus

    :return: A padded letter bigram model based on nltk.model.NgramModel
    """
    # subset the corpus to only include all-alpha tokens,
    # converted to lower-case (_after_ the all-alpha check)
    corpus_tokens = [word.lower() for word in corpus.words() if word.isalpha()]
    # Return the tokens and a smoothed (using the default estimator)
    #   padded bigram letter language model
    return LgramModel(2, corpus_tokens, True, True)


# Question 1.2 [7.5 marks]
def tweet_ent(file_name: str, bigram_model: LgramModel) -> List[Tuple[float, List[str]]]:
    """
    Using a character bigram model, compute sentence entropies
    for a subset of the tweet corpus, removing all non-alpha tokens and
    tweets with less than 5 all-alpha tokens, then converted to lowercase

    :param file_name: twitter file to process

    :return: ordered list of average entropies and tweets"""

    # Clean up the tweet corpus to remove all non-alpha
    # tokens and tweets with less than 5 (remaining) tokens, converted
    # to lowercase
    list_of_tweets = xtwc.sents(file_name)
    cleaned_list_of_tweets = [[word.lower() for word in tweet if word.isalpha()] for tweet in list_of_tweets]
    cleaned_list_of_tweets = [tweet for tweet in cleaned_list_of_tweets if len(tweet) >= 5]
    #print(len(cleaned_list_of_tweets))
    
    # Return a list of tuples of the form: (entropy,tweet)
    #  for each tweet in the cleaned corpus, where entropy is the
    #  average per_item bigram entropy of the tokens in the tweet.
    #  The list should be sorted by entropy.
    entropy_tweet_pairs = []
    for tweet in cleaned_list_of_tweets:
        sum_entropy = 0
        for word in tweet:
            entropy = bigram_model.entropy(word, False, True)
            sum_entropy += entropy
        average_entropy = sum_entropy / len(tweet)
        entropy_tweet_pairs.append((average_entropy, tweet))
    
    return sorted(entropy_tweet_pairs, key=lambda x : x[0])
    

# Question 1.3 [3 marks]
def short_answer_1_3() -> str:
    """
    Briefly explain what left and right padding accomplish and why
    they are a good idea. Assuming you have a bigram model trained on
    a large enough sample of English that all the relevant bigrams
    have reliable probability estimates, give an example of a string
    whose average letter entropy you would expect to be (correctly)
    greater with padding than without and explain why.
   
    :return: your answer
    """
    return inspect.cleandoc(
        """"
        Left and right padding adds tokens to signify the start and end of words which provides more context for the model, 
        allowing it to better understand the structure and placements of letters in a word. A string ending in “oq” will have a
        higher entropy with padding than without. Although the bigram is not unlikely, any word ending in “oq” will have a large entropy as 
        ‘q’ will rarely end a word.
        """
        )

# Question 1.4 [3 marks]
def short_answer_1_4() -> str:
    """
    Explain the output of lm.entropy('bbq',verbose=True,perItem=True)
    See the Coursework 1 instructions for details.

    :return: your answer
    """
    return inspect.cleandoc(
        """"
        p(b|('<s>',)) = [2-gram] 0.046511   Probability of the letter ‘b’ coming at the start of a word  
        p(b|('b',)) = [2-gram] 0.007750     Probability of the letter ‘b’ coming after the letter ‘b’
        backing off for ('b', 'q')          ‘q’ coming after the letter ‘b’ is never seen so you back off
        p(q|()) = [1-gram] 0.000892         Probability of seeing the letter ‘q’
        p(q|('b',)) = [2-gram] 0.000092     Probability of ‘q’ coming after the letter ‘b’
        p(</s>|('q',)) = [2-gram] 0.010636  Probability of the letter 'q' coming at the end of a word
        7.85102054894183                    Cross entropy
        """
        )

# Question 1.5 [3 marks]
def short_answer_1_5() -> str:
    """
    Inspect the distribution of tweet entropies and discuss.
    See the Coursework 1 instructions for details.

    :return: your answer
    """
    global ents
    # Uncomment the following lines when you are ready to work on this.
    # Please comment them out again or delete them before submitting.
    # Note that you will have to close the two plot windows to allow this
    #  function to return.
    #just_e = [e for (e,tw) in ents]
    #hist(just_e,"Bi-char entropies from cleaned twitter data")
    #hist(just_e,"Bi-char entropies from cleaned twitter data",
    #     log=True,block=True)
    return inspect.cleandoc(
        """
        There are two main collections in the plots: 
            - centred around the mean
            - high entropy skewed to the right

        Lower entropy tweets are likely in English since the model was trained on an English corpus and so it has more 
        confidence in its predictions. Tweets with high entropy are likely non-English. We can classify English and 
        non-English tweets using some entropy threshold which divides these classes.
        """
        )

# Question 1.6 [10 marks]
def is_English(bigram_model: LgramModel, tweet: List[str]) -> bool:
    """
    Classify if the given tweet is written in English or not.

    :param bigram_model: the bigram letter model trained on the Brown corpus
    :param tweet: the tweet
    :return: True if the tweet is classified as English, False otherwise
    """
    cleaned_tweet = [word.lower() for word in tweet if word.isalpha()]
    total_entropy = 0
    for word in cleaned_tweet:
        entropy = bigram_model.entropy(word, False, True)
        total_entropy += entropy
    average_entropy = total_entropy / len(cleaned_tweet)
    return average_entropy < 6


# Question 1.7 [16 marks]
def essay_question():
    """

    THIS IS AN ESSAY QUESTION WHICH IS INDEPENDENT OF THE PREVIOUS
    QUESTIONS ABOUT TWITTER DATA AND THE BROWN CORPUS!

    See the Coursework 1 instructions for a question about the average
    per word entropy of English.
    1) Name 3 problems that the question glosses over
    2) What kind of experiment would you perform to get a better estimate
       of the per word entropy of English?

    There is a limit of 400 words for this question.
    :return: your answer
    """
    return inspect.cleandoc("""Your answer""")


#############################################
# SECTION II - RESOLVING PP ATTACHMENT AMBIGUITY
#############################################

# Question 2.1 [15 marks]
class NaiveBayes:
    """
    Naive Bayes model with Lidstone smoothing (parameter alpha).
    """

    def __init__(self, data: List[Tuple[List[Any], str]], alpha: float):
        """
        :param data: A list with tuples of the form (list with features, label)
        :param alpha: \alpha value for Lidstone smoothing
        """
        self.vocab = self.get_vocab(data)
        self.alpha = alpha
        self.prior, self.likelihood = self.train(data, alpha, self.vocab)

    @staticmethod
    def get_vocab(data: List[Tuple[List[Any], str]]) -> Set[Any]:
        """
        Compute the set of all possible features from the (training) data.
        :param data: A list with tuples of the form (list with features, label)

        :return: The set of all features used in the training data for all classes.
        """
        feature_set = []
        for item in data:
            features = item[0]
            feature_set = feature_set + features        
        return set(feature_set)
        

    @staticmethod
    def train(data: List[Tuple[List[Any], str]], alpha: float, vocab: Set[Any]) -> Tuple[Dict[str, float],
          Dict[str, Dict[
          Any, float]]]:
        """
        Estimates the prior and likelihood from the data with Lidstone smoothing.

        :param data: A list of tuples ([f1, f2, ... ], c) with
                    the first element being a list of features and
                    the second element being its class.
        :param alpha: alpha value for Lidstone smoothing
        :param vocab: The set of all features used in the training data
                      for all classes.

        :return: Two dictionaries: the prior and the likelihood
                 (in that order).
        The returned values should relate as follows to the probabilities:
            prior[c] = P(c)
            likelihood[c][f] = P(f|c)
        """
        assert alpha >= 0.0
        
        total_number_training_documents = len(data)
        
        # Dict holds str:int where str is the class and int is the number of times the class appears in the data
        class_number_training_documents: Dict[str, int] = {}
        
        # Dict will hold all features that occur in a class
        features_in_class: Dict[str, List[Any]] = {}
                
        for entry in data:
            features = entry[0]
            class_ = entry[1]
            
            # Making dict with count of each class
            # If the class is not in dict yet then initialise its value to 1
            if class_ not in class_number_training_documents:
                class_number_training_documents[class_] = 1
            # Else class is in dict so add 1 to its total
            else: 
                class_number_training_documents[class_] += 1
            
            # If class not in dict then initialise features list
            if class_ not in features_in_class:
                features_in_class[class_] = features
            else:
                # If initialise then concat the new feature on to existing list
                features_in_class[class_] = features_in_class[class_] + features
                        
              
        # Compute prior (MLE). Compute likelihood with smoothing.
        prior = {}  
        for entry in data:
            class_ = entry[1]
            # Proportion of the training documents belonging to class c
            prior[class_] = (class_number_training_documents[class_])/(total_number_training_documents) 

        # Compute raw frequency distributions 
        likelihood = {} # key: str  val: Dict (key: Any  val: float)
                        # class                Feature     probability
        
        
        classes = set([entry[1] for entry in data])
        #print(f"Classes in train: {cla}")
        
        for class_ in classes:
            total_denominator_count = 0
            
            # First for loop finds the denominator in P(f|c):
            
            for feature in vocab:
                # Get count and alpha
                count = features_in_class[class_].count(feature) + alpha 
                # Add to denominator sum
                total_denominator_count += count 
            
            # Second for loop calculates P(f_i|c) for all i:
            
            # Holds probability for each feature in current class
            feature_probability = {}
            for feature in vocab:
                # Get count and alpha
                count = features_in_class[class_].count(feature) + alpha
                # Calculate P(f_i|c) and store ir
                feature_probability[feature] = count/total_denominator_count
            
            # Store whole dictionary for specific class
            likelihood[class_] = feature_probability     

        return prior, likelihood

    def prob_classify(self, d: List[Any]) -> Dict[str, float]:
        """
        Compute the probability P(c|d) for all classes.
        :param d: A list of features.

        :return: The probability p(c|d) for all classes as a dictionary.
        """
        class_prob: Dict[str, float] = {}
        classes = self.prior.keys()
        
        for class_ in classes:
            prior = self.prior[class_]
            # Likelihoods for current class
            likelihoods = self.likelihood[class_]
            # Calculating products
            likelihood_product = 1
            for feature in d:
                if feature in likelihoods:
                    feature_probability = likelihoods[feature]
                    likelihood_product *= feature_probability
            # Store final value under class name
            class_prob[class_] = prior*likelihood_product 
        
        # Sum all probabilities for all classes
        probabilities_sum = 0 
        for probability in class_prob.values():
            probabilities_sum += probability
        
        # Normalise each class
        for entry in class_prob:
            class_prob[entry] = class_prob[entry]/probabilities_sum
        
        return class_prob

    def classify(self, d: List[Any]) -> str:
        """
        Compute the most likely class of the given "document" with ties broken arbitrarily.
        :param d: A list of features.

        :return: The most likely class.
        """
        class_prob = self.prob_classify(d)
        return max(class_prob, key=class_prob.get)


# Question 2.2 [15 marks]
def open_question_2_2() -> str:
    """
    See the Coursework 1 instructions for detail of the following:
    1) The differences in accuracy between the different ways
        to extract features?
    2) The difference between Naive Bayes vs Logistic Regression
    3) An explanation of a binary feature that returns 1
        if V=`imposed' AND N_1 = `ban' AND P=`on' AND N_2 = `uses'.

    Limit: 150 words for all three sub-questions together.
    """
    return inspect.cleandoc("""Your answer""")

# Feature extractors used in the table:

def feature_extractor_1(v: str, n1: str, p: str, n2: str) -> List[Any]:
    return [("v", v)]

def feature_extractor_2(v: str, n1: str, p: str, n2: str) -> List[Any]:
    return [("n1", n1)]

def feature_extractor_3(v: str, n1: str, p: str, n2: str) -> List[Any]:
    return [("p", p)]

def feature_extractor_4(v: str, n1: str, p: str, n2: str) -> List[Any]:
    return [("n2", n2)]

def feature_extractor_5(v: str, n1: str, p: str, n2: str) -> List[Any]:
    return [("v", v), ("n1", n1), ("p", p), ("n2", n2)]


# Question 2.3, part 1 [10 marks]
def your_feature_extractor(v: str, n1: str, p:str, n2:str) -> List[Any]:
    """
    Takes the head words and produces a list of features. The features may
    be of any type as long as they are hashable.

    :param v: The verb.
    :param n1: Head of the object NP.
    :param p: The preposition.
    :param n2: Head of the NP embedded in the PP.

    :return: A list of features produced by you.
    """
    return [("v",v), ("n1",n1), ("p",p), ("n2",n2), ("vn1", (v,n1)), ("vp", (v,p)), ("vn2", (v,n2)), ("n1p", (n1,p)), ("n1n2", (n1,n2)), ("pn2", (p,n2))]
    
# n1pn2

# Question 2.3, part 2 [10 marks]
def open_question_2_3() -> str:
    """
    Briefly describe your feature templates and your reasoning for them.
    Pick three examples of informative features and discuss why they make sense or why they do not make sense
    and why you think the model relies on them.

    There is a limit of 300 words for this question.
    """
    return inspect.cleandoc("""Your answer""")


"""
Format the output of your submission for both development and automarking. 
!!!!! DO NOT MODIFY THIS PART !!!!!
"""

def answers():
    # Global variables for answers that will be used by automarker
    global ents, lm, top10_ents, bottom10_ents
    global answer_open_question_2_2, answer_open_question_2_3
    global answer_short_1_4, answer_short_1_5, answer_short_1_3, answer_essay_question

    global naive_bayes
    global acc_extractor_1, naive_bayes_acc, lr_acc, logistic_regression_model, dev_features
    global dev_tweets_preds


    print("*** Part I***\n")
    
    print("*** Question 1.1 ***")
    print('Building Brown news bigram letter model ... ')
    lm = train_LM(brown)
    print('Letter model built')

    print("*** Question 1.2 ***")
    ents = tweet_ent(twitter_file_ids, lm)

    top10_ents = ents[:10]
    bottom10_ents = ents[-10:]

    answer_short_1_3 = short_answer_1_3()
    print("*** Question 1.3 ***")
    print(answer_short_1_3)

    answer_short_1_4 = short_answer_1_4()
    print("*** Question 1.4 ***")
    print(answer_short_1_4)

    answer_short_1_5 = short_answer_1_5()
    print("*** Question 1.5 ***")
    print(answer_short_1_5)

    print("*** Question 1.6 ***")
    all_dev_ok = True
    dev_tweets_preds = []
    for tweet, gold_answer in get_annotated_tweets():
        prediction = is_English(lm, tweet)
        dev_tweets_preds.append(prediction)
        if prediction != gold_answer:
            all_dev_ok = False
            print("Missclassified", tweet)
    if all_dev_ok:
        print("All development examples correctly classified! "
              "We encourage you to test and tweak your classifier on more tweets.")

    answer_essay_question = essay_question()
    print("*** Question 1.7 (essay question) ***")
    print(answer_essay_question)
    

    print("*** Part II***\n")
    """"
    print("*** Question 2.1 ***")
    naive_bayes = NaiveBayes(apply_extractor(feature_extractor_5, ppattach.tuples("training")), 0.1)
    naive_bayes_acc = compute_accuracy(naive_bayes, apply_extractor(feature_extractor_5, ppattach.tuples("devset")))
    print(f"Accuracy on the devset: {naive_bayes_acc * 100}%")

    print("*** Question 2.2 ***")
    answer_open_question_2_2 = open_question_2_2()
    print(answer_open_question_2_2)

    print("*** Question 2.3 ***")
    training_features = apply_extractor(your_feature_extractor, ppattach.tuples("training"))
    dev_features = apply_extractor(your_feature_extractor, ppattach.tuples("devset"))
    logistic_regression_model = NltkClassifierWrapper(MaxentClassifier, training_features, max_iter=10)
    lr_acc = compute_accuracy(logistic_regression_model, dev_features)

    print("30 features with highest absolute weights")
    logistic_regression_model.show_most_informative_features(30)

    print(f"Accuracy on the devset: {lr_acc*100}")

    answer_open_question_2_3 = open_question_2_3()
    print("Answer to open question:")
    print(answer_open_question_2_3)
    """



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--answers':
        from autodrive_embed import run, carefulBind
        import adrive1

        with open("userErrs.txt", "w") as errlog:
            run(globals(), answers, adrive1.extract_answers, errlog)
    else:
        answers()
