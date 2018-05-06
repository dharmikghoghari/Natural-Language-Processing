import json
import perceplearn3
import time
import sys

def get_clean_sentence(sentence):
    new_sentence = ""
    for c in sentence:
        if c in '><?."()|-#*+;!/\\=,:$[]{}@^&~%': #removed single quote
            c = ' '
        new_sentence += c
    return new_sentence

def read_file_vanilla():
    f = [json.loads(x) for x in open("vanillamodel.txt", "r", encoding='UTF-8').read().split("\n")]
    #f = [json.loads(x) for x in open(sys.argv[-2], "r", encoding='UTF-8').read().split("\n")]

    weights_pn = f[0]
    weights_ft = f[1]
    bias_pn = f[2]
    bias_ft = f[3]

    return weights_pn, weights_ft, bias_pn, bias_ft

def get_word_count(actual_words):
    counts_dict = dict()
    for word in actual_words:
        if word not in counts_dict:
            counts_dict[word] = 1
        else:
            counts_dict[word] += 1

    return counts_dict

def read_file_averaged():
    f = [json.loads(x) for x in open("averagedmodel.txt", "r", encoding='UTF-8').read().split("\n")]

    avg_weights_pn = f[0]
    avg_weights_ft = f[1]
    avg_bias_pn = f[2]
    avg_bias_ft = f[3]

    return avg_weights_pn, avg_weights_ft, avg_bias_pn, avg_bias_ft

def write_to_file(answer_tags):
    fh = open("percepoutput.txt", "w+", encoding="UTF-8")
    for line in answer_tags:
        fh.write(line)
        fh.write("\n")
    fh.close()

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

def read_test_file():
    f = open("test-text.txt", "r", encoding='UTF-8')  # consider encoding UTF-8
    #f = open(sys.argv[-1], "r", encoding='UTF-8')  # consider encoding UTF-8
    tf = f.read()
    test_data = tf.splitlines()
    return test_data

def perform_classify(weights_pn, weights_ft, bias_pn, bias_ft, test_data):
    answer_tags = []
    for line in test_data:
        pos = line.find(' ')
        key = line[0:pos]
        sentence = line[pos:]

        sentence = sentence.lower()
        new_line = get_clean_sentence(sentence)
        all_words = new_line.split()

        activation_pn = 0
        activation_ft = 0

        for word in all_words:
            if word in weights_pn:
                activation_pn += weights_pn[word]
            if word in weights_ft:
                activation_ft += weights_ft[word]
        activation_pn += bias_pn
        activation_ft += bias_ft

        if activation_ft >= 0:
            f_t = 'Fake'
        else:
            f_t = 'True'

        if activation_pn >= 0:
            p_n = 'Neg'
        else:
            p_n = 'Pos'

        temp = key + " " + f_t + " " + p_n

        answer_tags.append(temp)

    return answer_tags

def main():
    test_data = read_test_file()

    weights_pn, weights_ft, bias_pn, bias_ft = read_file_vanilla()
    answer_tags = perform_classify(weights_pn, weights_ft, bias_pn, bias_ft, test_data)
    write_to_file(answer_tags)
    f1_score = calcF1("test-key.txt", "percepoutput.txt")
    print("f1_score = ", f1_score)
    print()

    avg_weights_pn, avg_weights_ft, avg_bias_pn, avg_bias_ft = read_file_averaged()
    answer_tags = perform_classify(avg_weights_pn, avg_weights_ft, avg_bias_pn, avg_bias_ft, test_data)
    write_to_file(answer_tags)
    f1_score = calcF1("test-key.txt", "percepoutput.txt")
    print("f1_score = ", f1_score)

# if __name__ == '__main__':
#     main()

if __name__ == '__main__':
    t = time.time()
    perceplearn3.main()
    main()
    print("time taken = ", time.time() - t)