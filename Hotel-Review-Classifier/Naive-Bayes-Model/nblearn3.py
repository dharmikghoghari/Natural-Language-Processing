import math
import sys
import re
import json

def get_clean_sentence(sentence):
    new_sentence = ""
    for c in sentence:
        if c in '><?."()|-#*+;!/\\=,:$[]{}@^&~%': #removed single quote
            c = " "
        new_sentence += c
    return new_sentence

def print_results(params):
    fake_total = params['fake_total']
    true_total = params['true_total']
    pos_total = params['pos_total']
    neg_total = params['neg_total']

    fake_class = params['fake_class']
    true_class = params['true_class']
    pos_class = params['pos_class']
    neg_class = params['neg_class']

    fake_word_count_total = params['fake_word_count_total']
    true_word_count_total = params['true_word_count_total']
    pos_word_count_total = params['pos_word_count_total']
    neg_word_count_total = params['neg_word_count_total']

    vocab = params['vocab']

    print("Fake Total =", fake_total)
    print("True Total =", true_total)
    print("Pos Total =", pos_total)
    print("Neg Total =", neg_total)

    print("Fake Class =", len(fake_class))
    print("True Class =", len(true_class))
    print("Pos Class =", len(pos_class))
    print("Neg Class =", len(neg_class))

    print("fake_word_count_total", fake_word_count_total)
    print("true_word_count_total", true_word_count_total)
    print("pos_word_count_total", pos_word_count_total)
    print("neg_word_count_total", neg_word_count_total)

    print("Vocab Size =", len(vocab))
    # print("Vocab = ", vocab)

def read_file():
    f = open("dev-train-labeled.txt", "r", encoding='UTF-8')
    t = f.read()
    data = t.splitlines()
    total_sentences = len(data)

    params = dict()

    fake_class = dict()
    true_class = dict()
    pos_class = dict()
    neg_class = dict()
    vocab = set()

    fake_total = 0
    true_total = 0
    pos_total = 0
    neg_total = 0

    fake_word_count_total = 0
    true_word_count_total = 0
    pos_word_count_total = 0
    neg_word_count_total = 0

    for line in data:
        sentence = line[8:]
        sentence = sentence.lower()
        new_sentence = get_clean_sentence(sentence)
        all_words = new_sentence.split()

        f_or_t = all_words[0]
        p_or_n = all_words[1]
        actual_words = all_words[2:]

        #common_words = ['on', 'after', 'while', 'further', 'below', 'with', "i'd", 'nor', "he'd", 'me', "you're", 'and', 'up', "you'll", 'to', 'why', 'his', "they've", 'both', 'him', "we'd", 'how', 'am', 'from', 'her', 'was', 'its', 'before', 'them', 'there', 'be', 'ought', 'or', "why's", 'does', "how's", 'very', "where's", 'high', 'about', "what's", "i'm", 'most', 'whom', 'yours', 'the', "he'll", "it's", 'were', 'is', "you'd", "i've", 'own', 'for', "we've", 'having', 'other', 'which', 'between', 'doing', "that's", 'their', "you've", 'have', 'are', 'he', 'at', 'than', "we're", "they'd", 'of', 'once', 'when', 'out', 'myself', 'our', 'yourself', 'here', 'my', "she'd", 'only', 'she', 'in', 'where', 'ours', 'until', 'an', 'because', 'also', 'do', 'theirs', 'you', "we'll", 'hers', 'himself', "she'll", 'herself', "they'll", 'some', 'been', "there's", 'down', "here's", 'again', 'being', 'these', 'yourselves', 'each', 'i', 'now', "she's", 'who', 'those', 'if', 'during', 'all', 'we', 'they', 'but', 'as', "when's", 'it', 'so', "let's", 'this', 'few', 'themselves', 'into', 'against', 'a', 'what', "they're", 'above', 'ourselves', 'had', 'by', 'through', "who's", 'should', 'your', 'that', 'could', 'then', 'under', 'over', 'same', 'such', 'did', 'more', 'would', 'itself', "i'll", 'too', 'any', "he's", 'has']

        common_words = ['old', 'its', 'or', 'next', 'through', 'our', "he'll", 'yours', 'when', 'few', 'is', 'each', 'once', 'i', 'at', 'the', 'are', 'having', "she's", 'where', 'you', 'in', 'why', 'that', 'because', 'ourselves', "here's", "they'd", 'did', 'yourselves', "she'll", 'could', "they're", 'out', 'has', "i'm", 'he', 'other', "we'll", "you're", 'so', 'their', 'who', 'if', 'am', 'myself', 'many', 'own', 'very', 'would', 'there', 'from', 'on', 'both', 'such', 'above', "that's", "we've", 'any', 'doing', "why's", 'him', "when's", 'all', "there's", "they'll", 'about', 'which', 'quite', 'ought', 'before', "she'd", 'herself', 'every', 'itself', 'away', 'my', "where's", 'but', 'as', 'some', 'for', 'here', 'we', 'being', 'do', 'and', 'most', "i've", 'what', 'her', 'your', "you've", 'ours', 'over', 'should', 'again', 'had', 'high', 'make', 'his', 'this', 'me', 'nor', 'of', 'into', 'by', 'below', 'were', "it's", 'now', "let's", 'it', "you'll", 'these', 'only', "they've", 'she', "i'll", 'said', "how's", 'too', 'also', 'they', "what's", 'between', 'up', 'against', 'themselves', "i'd", 'those', 'while', 'be', 'with', 'though', 'them', 'theirs', 'then', 'after', 'to', 'himself', 'under', 'does', 'an', 'asked', 'further', 'rooms', 'was', "you'd", "who's", 'down', 'whom', "he's", 'have', 'been', 'until', 'same', 'how', 'a', 'more', 'during', 'than', "he'd", "we're", "we'd", 'yourself', 'two', 'hers']

        #Fake_class
        if f_or_t == "fake":
            fake_total += 1
            count_of_words = 0
            for word in actual_words:
                if word not in common_words:
                    count_of_words += 1
                    vocab.add(word)
                    if word not in fake_class:
                        fake_class[word] = 1
                    else:
                        fake_class[word] += 1
            fake_word_count_total += count_of_words

       #True_class
        else:
            true_total += 1
            count_of_words = 0
            for word in actual_words:
                if word not in common_words:
                    count_of_words += 1
                    vocab.add(word)
                    if word not in true_class:
                        true_class[word] = 1
                    else:
                        true_class[word] += 1
            true_word_count_total += count_of_words

        #Pos_class
        if p_or_n == "pos":
            pos_total += 1
            count_of_words = 0
            for word in actual_words:
                if word not in common_words:
                    count_of_words += 1
                    vocab.add(word)
                    if word not in pos_class:
                        pos_class[word] = 1
                    else:
                        pos_class[word] += 1
            pos_word_count_total += count_of_words

        #Neg_class
        else:
            neg_total += 1
            count_of_words = 0
            for word in actual_words:
                if word not in common_words:
                    count_of_words += 1
                    vocab.add(word)
                    if word not in neg_class:
                        neg_class[word] = 1
                    else:
                        neg_class[word] += 1
            neg_word_count_total += count_of_words

    params['fake_total'] = fake_total
    params['true_total'] = true_total
    params['pos_total'] = pos_total
    params['neg_total'] = neg_total

    params['fake_class'] = fake_class
    params['true_class'] = true_class
    params['pos_class'] = pos_class
    params['neg_class'] = neg_class

    params['fake_word_count_total'] = fake_word_count_total
    params['true_word_count_total'] = true_word_count_total
    params['pos_word_count_total'] = pos_word_count_total
    params['neg_word_count_total'] = neg_word_count_total

    params['vocab'] = vocab
    params['total_sentences'] = total_sentences
    return params

def smoothing(params):
    fake_class = params['fake_class']
    true_class = params['true_class']
    pos_class = params['pos_class']
    neg_class = params['neg_class']

    vocab = params['vocab']

    fake_word_count_total = params['fake_word_count_total'] + len(vocab)       #clarify this
    true_word_count_total = params['true_word_count_total'] + len(vocab)
    pos_word_count_total = params['pos_word_count_total'] + len(vocab)
    neg_word_count_total = params['neg_word_count_total'] + len(vocab)

    factor = 1
    for item in vocab:
        if item not in fake_class:
            fake_class[item] = factor
        else:
            fake_class[item] += factor
        fake_class[item] = math.log((1.0 * fake_class[item]) / (1.0 * fake_word_count_total))

    for item in vocab:
        if item not in true_class:
            true_class[item] = factor
        else:
            true_class[item] += factor
        true_class[item] = math.log((1.0 * true_class[item]) / (1.0 * true_word_count_total))

    for item in vocab:
        if item not in pos_class:
            pos_class[item] = factor
        else:
            pos_class[item] += factor
        pos_class[item] = math.log((1.0 * pos_class[item]) / (1.0 * pos_word_count_total))

    for item in vocab:
        if item not in neg_class:
            neg_class[item] = factor
        else:
            neg_class[item] += factor
        neg_class[item] = math.log((1.0 * neg_class[item]) / (1.0 * neg_word_count_total))

    # print(fake_class)
    # sys.exit(0)

    params['fake_class'] = fake_class
    params['true_class'] = true_class
    params['pos_class'] = pos_class
    params['neg_class'] = neg_class

    params['fake_word_count_total'] = fake_word_count_total
    params['true_word_count_total'] = true_word_count_total
    params['pos_word_count_total'] = pos_word_count_total
    params['neg_word_count_total'] = neg_word_count_total

    return params

def write_to_file(params):
    fh = open("nbmodel.txt", "w+", encoding="UTF-8")
    fh.write(json.dumps(params['fake_total']))
    fh.write("\n")
    fh.write(json.dumps(params['true_total']))
    fh.write("\n")
    fh.write(json.dumps(params['pos_total']))
    fh.write("\n")
    fh.write(json.dumps(params['neg_total']))
    fh.write("\n")

    fh.write(json.dumps(params['fake_class']))
    fh.write("\n")
    fh.write(json.dumps(params['true_class']))
    fh.write("\n")
    fh.write(json.dumps(params['pos_class']))
    fh.write("\n")
    fh.write(json.dumps(params['neg_class']))

    fh.write("\n")
    fh.write(json.dumps(params['fake_word_count_total']))
    fh.write("\n")
    fh.write(json.dumps(params['true_word_count_total']))
    fh.write("\n")
    fh.write(json.dumps(params['pos_word_count_total']))
    fh.write("\n")
    fh.write(json.dumps(params['neg_word_count_total']))

    # fh.write("\n")
    # fh.write(params['vocab'])

    fh.write("\n")
    fh.write(json.dumps(params['total_sentences']))

    fh.close()

def main():
    params = read_file()
    params = smoothing(params)
    write_to_file(params)

    print_results(params)
    #print((params['fake_class']))

#main()