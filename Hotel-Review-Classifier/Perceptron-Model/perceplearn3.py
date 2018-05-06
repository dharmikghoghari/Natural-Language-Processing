import math
import sys
import re
import time
import json

def get_clean_sentence(sentence):
    new_sentence = ""
    for c in sentence:
        if c in '><?."()|-#*+;!/\\=,:$[]{}@^&~%':  #removed single quote
            c = " "
        new_sentence += c
    return new_sentence

def get_word_count(actual_words, common_words):
    counts_dict = dict()
    for word in actual_words:
        if word not in common_words:
            if word not in counts_dict:
                counts_dict[word] = 1
            else:
                counts_dict[word] += 1

    return counts_dict

def read_file():
    #f = open(sys.argv[-1], "r", encoding='UTF-8')
    f = open("dev-train-labeled.txt", "r", encoding='UTF-8')
    t = f.read()
    data = t.splitlines()
    data_word_count = []

    #common parameters for averaged and vanilla perceptron
    weights_pn = dict()
    weights_ft = dict()
    bias_pn = 0
    bias_ft = 0

    #parameters for aceraged perceptron
    mew_ft = dict()
    mew_pn = dict()
    beta_ft = 0
    beta_pn = 0
    c = 1

    max_iterations = 30

    common_words = [
        "a", "about", "above", "across", "after", "again", "all", "almost",
        "along", "also", "although", "always", "am", "amount", "an", "and",
        "another", "any", "anyone", "anything", "anyway", "are", "around", "as", "at", "back",
        "because", "been", "before", "being",
        "between", "beyond", "both", "bottom", "but", "by", "call", "can"
        , "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "during",
        "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every",
        "everyone", "everywhere", "few", "fifteen", "fify", "fill", "find", "fire", "first",
        "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get",
        "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "hereafter", "hereby", "herein", "hereupon",
        "hers", "herself", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed",
        "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd",
        "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
        "move", "much", "must", "my", "name", "namely", "neither", "never", "nevertheless", "next", "nine",
        "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once",
        "one", "only", "onto", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own",
        "part", "per", "perhaps", "please", "put", "rather", "same", "see", "seem", "seemed", "seeming", "seems",
        "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
        "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "system", "take", "ten", "than",
        "that", "the", "their", "them", "then", "thence", "there", "thereafter", "thereby", "therefore",
        "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three",
        "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two",
        "un", "under", "until", "up", "upon", "us", "very", "was", "we", "well", "were", "what", "whatever", "when",
        "whence", "whenever", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever",
        "whether", "which", "while", "whither", "whoever", "whole", "whom", "whose", "why", "will", "with", "within",
        "without", "would", "yet", "you", "your", "yours", "yourselves", "afterwards" , "against", "alone", "already",
        "among", "amongst", "amoungst", "anyhow","be","became", "become"
        , "becomes", "becoming",  "beforehand", "behind", "below", "beside",  "besides", "bill", "cannot",
        "cant", "anywhere", "everything"]

    t = time.time()
    #first iteration along with preprocessing
    for sentence in data:
        sentence = sentence[8:]
        sentence = sentence.lower()
        sentence = get_clean_sentence(sentence)
        all_words = sentence.split()

        current_line_stats = []

        if all_words[0] == 'true':  # first word is either true or fake
            y_ft = -1
        else:
            y_ft = 1
        if all_words[1] == 'pos':   # second word is either pos or neg
            y_pn = -1
        else:
            y_pn = 1

        current_line_stats.append(y_ft)
        current_line_stats.append(y_pn)

        actual_words = all_words[2:]
        current_line_word_count = get_word_count(actual_words, common_words)
        current_line_stats.append(current_line_word_count)
        data_word_count.append(current_line_stats)

        activation_ft = 0
        activation_pn = 0

        # compute activation
        for word in current_line_word_count:
            if word not in common_words:
                if word not in weights_pn:
                    mew_pn[word] = weights_pn[word] = 0
                else:
                    activation_pn += weights_pn[word] * current_line_word_count[word]

                if word not in weights_ft:
                    mew_ft[word] = weights_ft[word] = 0
                else:
                    activation_ft += weights_ft[word] * current_line_word_count[word]

        activation_pn += bias_pn
        activation_ft += bias_ft

        # update weights and bias for pos neg classifier perceptron if a condition is met
        if y_pn * activation_pn <= 0:
            for word in current_line_word_count:
                weights_pn[word] += y_pn * current_line_word_count[word]
                mew_pn[word] += y_pn * c * current_line_word_count[word]
            bias_pn += y_pn
            beta_pn += y_pn * c

        # update weights and bias for fake true classifier perceptron if a condition is met
        if y_ft * activation_ft <= 0:
            for word in current_line_word_count:
                weights_ft[word] += y_ft * current_line_word_count[word]
                mew_ft[word] += y_ft * c * current_line_word_count[word]
            bias_ft += y_ft
            beta_ft += y_ft * c

        c += 1
    print("Time taken :", time.time() - t)

    #next iterations
    for i in range(0, max_iterations - 1):
        for line_data in data_word_count:

            y_ft = line_data[0]
            y_pn = line_data[1]
            current_line_word_count = line_data[2]

            activation_ft = 0
            activation_pn = 0

            # compute activation
            for word in current_line_word_count:
                activation_pn += weights_pn[word] * current_line_word_count[word]
                activation_ft += weights_ft[word] * current_line_word_count[word]

            activation_pn += bias_pn
            activation_ft += bias_ft

            # update weights and bias for pos neg classifier perceptron
            if y_pn * activation_pn <= 0:
                for word in current_line_word_count:
                    weights_pn[word] += y_pn * current_line_word_count[word]
                    mew_pn[word] += y_pn * c * current_line_word_count[word]
                bias_pn += y_pn
                beta_pn += y_pn * c

            # update weights and bias for fake true classifier perceptron
            if y_ft * activation_ft <= 0:
                for word in current_line_word_count:
                    weights_ft[word] += y_ft * current_line_word_count[word]
                    mew_ft[word] += y_ft * c * current_line_word_count[word]
                bias_ft += y_ft
                beta_ft += y_ft * c
            c += 1

    return weights_pn, weights_ft, bias_pn, bias_ft, mew_pn, mew_ft, beta_pn, beta_ft, c

def write_to_file(weights_pn, weights_ft, bias_pn, bias_ft, mew_pn, mew_ft, beta_pn, beta_ft, c):
    f1 = open("vanillamodel.txt", "w+", encoding="UTF-8")
    f1.write(json.dumps(weights_pn))
    f1.write("\n")
    f1.write(json.dumps(weights_ft))
    f1.write("\n")
    f1.write(json.dumps(bias_pn))
    f1.write("\n")
    f1.write(json.dumps(bias_ft))
    f1.close()

    for item in weights_pn:
        weights_pn[item] -= (mew_pn[item]/c)
    for item in weights_ft:
        weights_ft[item] -= (mew_ft[item]/c)

    bias_pn -= (beta_pn/c)
    bias_ft -= (beta_ft/c)

    f2 = open("averagedmodel.txt", "w+", encoding="UTF-8")
    f2.write(json.dumps(weights_pn))
    f2.write("\n")
    f2.write(json.dumps(weights_ft))
    f2.write("\n")
    f2.write(json.dumps(bias_pn))
    f2.write("\n")
    f2.write(json.dumps(bias_ft))
    f2.close()

def main():

    weights_pn, weights_ft, bias_pn, bias_ft, mew_pn, mew_ft, beta_pn, beta_ft, c = read_file()
    write_to_file(weights_pn, weights_ft, bias_pn, bias_ft, mew_pn, mew_ft, beta_pn, beta_ft, c)

if __name__ == '__main__':
    main()