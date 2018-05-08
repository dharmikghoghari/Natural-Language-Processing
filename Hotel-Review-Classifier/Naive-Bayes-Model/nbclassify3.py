import json
import sys
import math
import nblearn3
import time

def get_clean_sentence(sentence):
    new_sentence = ""
    for c in sentence:
        if c in '><?."()|-#*+;!/\\=,:$[]{}@^&~%': #removed single quote
            c = ' '
        new_sentence += c
    return new_sentence

def read_file():
    f = [json.loads(x) for x in open("nbmodel.txt", "r", encoding='UTF-8').read().split("\n")]

    fake_total = f[0]
    true_total = f[1]
    pos_total = f[2]
    neg_total = f[3]

    fake_class = f[4]
    true_class = f[5]
    pos_class = f[6]
    neg_class = f[7]

    fake_word_count_total = f[8]
    true_word_count_total = f[9]
    pos_word_count_total = f[10]
    neg_word_count_total = f[11]

    total_sentences = f[12]

    params = dict()

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

    params['total_sentences'] = total_sentences

    return params

def read_test_file():
    f = open("test-text.txt", "r", encoding='UTF-8')  # consider encoding UTF-8
    # f = open(sys.argv[-1], "r", encoding='UTF-8')  # consider encoding UTF-8
    tf = f.read()
    test_data = tf.splitlines()
    return test_data

def write_to_file(answer_tags):
    fh = open("nboutput.txt", "w+", encoding="UTF-8")
    for line in answer_tags:
        fh.write(line)
        fh.write("\n")
    fh.close()

def classify(key, params, words, priors):
    fake_or_true = ""
    pos_or_neg = ""

    fake_class = params['fake_class']
    true_class = params['true_class']
    pos_class = params['pos_class']
    neg_class = params['neg_class']

    fake_prior = priors['fake_prior']
    true_prior = priors['true_prior']
    pos_prior = priors['pos_prior']
    neg_prior = priors['neg_prior']

    fake_prob = fake_prior
    true_prob = true_prior
    pos_prob = pos_prior
    neg_prob = neg_prior

    for word in words:
        if word in fake_class:
            fake_prob += fake_class[word]
        if word in true_class:
            true_prob += true_class[word]
        if word in pos_class:
            pos_prob += pos_class[word]
        if word in neg_class:
            neg_prob += neg_class[word]

    if fake_prob < true_prob:
        fake_or_true = "True"
    else:
        fake_or_true = "Fake"

    if pos_prob > neg_prob:
        pos_or_neg = "Pos"
    else:
        pos_or_neg = "Neg"

    output = key + " " + fake_or_true + " " + pos_or_neg
    return output

def perform_classify(params, test_data):
    answer_tags = []

    priors = dict()
    priors['fake_prior'] = math.log(1.0 * params['fake_total'] / params['total_sentences'])
    priors['true_prior'] = math.log(1.0 * params['true_total'] / params['total_sentences'])
    priors['pos_prior'] = math.log(1.0 * params['pos_total'] / params['total_sentences'])
    priors['neg_prior'] = math.log(1.0 * params['neg_total'] / params['total_sentences'])

    for line in test_data:
        pos = line.find(' ')
        key = line[0:pos]
        sentence = line[pos:]

        sentence = sentence.lower()
        new_line = get_clean_sentence(sentence)
        all_words = new_line.split()

        output = classify(key, params, all_words, priors)
        answer_tags.append(output)
    return answer_tags

def calcF1(true_key_path, pred_key_path):

    acc_count = 0

    file_orig = open(true_key_path, "r")
    true_keys = file_orig.read().splitlines()

    file_pred = open(pred_key_path, "r")
    pred_keys = file_pred.read().splitlines()

    t_f_class_counts = [0,0,0,0]
    p_n_class_counts = [0,0,0,0]

    for i in range(0, len(true_keys)):

        true_key_entry = true_keys[i]
        pred_key_entry = pred_keys[i]

        if true_key_entry == pred_key_entry:
            acc_count = acc_count + 1

        true_key_entry_parts = true_key_entry.split(" ")
        pred_key_entry_parts = pred_key_entry.split(" ")

        if true_key_entry_parts[0] == pred_key_entry_parts[0]:

            true_label_t_f = true_key_entry_parts[1]
            pred_label_t_f = pred_key_entry_parts[1]

            true_label_p_n = true_key_entry_parts[2]
            pred_label_p_n = pred_key_entry_parts[2]

            if true_label_t_f == "True" and pred_label_t_f == "True":
                t_f_class_counts[0] = t_f_class_counts[0] + 1
            elif true_label_t_f == "Fake" and pred_label_t_f == "True":
                t_f_class_counts[1] = t_f_class_counts[1] + 1
            elif true_label_t_f == "True" and pred_label_t_f == "Fake":
                t_f_class_counts[2] = t_f_class_counts[2] + 1
            elif true_label_t_f == "Fake" and pred_label_t_f == "Fake":
                t_f_class_counts[3] = t_f_class_counts[3] + 1

            if true_label_p_n == "Pos" and pred_label_p_n == "Pos":
                p_n_class_counts[0] = p_n_class_counts[0] + 1
            elif true_label_p_n == "Neg" and pred_label_p_n == "Pos":
                p_n_class_counts[1] = p_n_class_counts[1] + 1
            elif true_label_p_n == "Pos" and pred_label_p_n == "Neg":
                p_n_class_counts[2] = p_n_class_counts[2] + 1
            elif true_label_p_n == "Neg" and pred_label_p_n == "Neg":
                p_n_class_counts[3] = p_n_class_counts[3] + 1

    prec_true = (t_f_class_counts[0] * 1.0) / (t_f_class_counts[0]+t_f_class_counts[1])*1.0
    rec_true = (t_f_class_counts[0] * 1.0) / (t_f_class_counts[0] + t_f_class_counts[2]) * 1.0
    prec_fake = (t_f_class_counts[3] * 1.0) / (t_f_class_counts[3] + t_f_class_counts[2]) * 1.0
    rec_fake = (t_f_class_counts[3] * 1.0) / (t_f_class_counts[3] + t_f_class_counts[1]) * 1.0

    f_true = (2.0*prec_true*rec_true)/(prec_true+rec_true)*1.0
    f_fake = (2.0 * prec_fake * rec_fake) / (prec_fake + rec_fake) * 1.0

    prec_pos = (p_n_class_counts[0] * 1.0) / (p_n_class_counts[0] + p_n_class_counts[1]) * 1.0
    rec_pos = (p_n_class_counts[0] * 1.0) / (p_n_class_counts[0] + p_n_class_counts[2]) * 1.0
    prec_neg = (p_n_class_counts[3] * 1.0) / (p_n_class_counts[3] + p_n_class_counts[2]) * 1.0
    rec_neg = (p_n_class_counts[3] * 1.0) / (p_n_class_counts[3] + p_n_class_counts[1]) * 1.0

    f_pos = (2.0 * prec_pos * rec_pos) / (prec_pos + rec_pos) * 1.0
    f_neg = (2.0 * prec_neg * rec_neg) / (prec_neg + rec_neg) * 1.0

    f1_score = 1.0*(f_true + f_fake +f_pos +f_neg)/ 4.0

    print("True class f1 : ", f_true)
    print("Fake class f1 : ", f_fake)
    print("Pos class f1 : ", f_pos)
    print("Neg class f1 : ", f_neg)

    print("Combine accuracy : ",acc_count/len(true_keys))
    acc1 = (p_n_class_counts[0]+p_n_class_counts[3])/(p_n_class_counts[0]+p_n_class_counts[1]+p_n_class_counts[2]+p_n_class_counts[3])
    acc2 = (t_f_class_counts[0]+t_f_class_counts[3])/(t_f_class_counts[0]+t_f_class_counts[1]+t_f_class_counts[2]+t_f_class_counts[3])

    print("True/Fake Accuracy : ", acc2)
    print("Pos/Neg Accuracy : ", acc1)

    print("Avg accuracy : ", (acc2+acc1)/2)

    return f1_score

def main():
    params = read_file()
    test_data = read_test_file()
    answer_tags = perform_classify(params, test_data)
    write_to_file(answer_tags)
    #calc_f1_measure()
    f1_score = calcF1("test-key.txt", "nboutput.txt")
    print("f1_score = ", f1_score)

#main()

if __name__ == '__main__':
    t = time.time()
    nblearn3.main()
    main()
    print("time taken = ", time.time() - t)
