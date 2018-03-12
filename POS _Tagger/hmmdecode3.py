import hmmlearn3
import json
import sys
import time

def read_file():
    #tags = dict()
    word_given_tag = dict()
    tag_given_pervtag = dict()

    f = [json.loads(x) for x in open("hmmmodel.txt", "r", encoding='UTF-8').read().split("\n")]# consider encoding UTF-8
    #data = json.load(f)
    #tags = data[0]                      #count of tags stored in the 3rd line in hmm model
    tag_given_pervtag = f[0]         # count of tag-->nextTag stored in the 7th line in hmm model
    word_given_tag = f[1]
    tags = f[2]                     # count of word/tag stored in the 11th line in the hmm model
    #prev_total_sentences = data[3]      # count of total number of sentences, will be used in initialization step of viterbi
    return tag_given_pervtag, word_given_tag

def read_test_file():
    f = open("en_dev_raw.txt", "r", encoding='UTF-8')  # consider encoding UTF-8
    #f = open(sys.argv[-1], "r", encoding='UTF-8')  # consider encoding UTF-8
    tf = f.read()
    test_data = tf.splitlines()
    return test_data

def getMaxProb(prev_dict, word, curr_tag, tag_given_prevtag, word_given_tag):
    maxi = 0
    pointer = ""
    if word in word_given_tag:
        for tag in prev_dict:
            previous_state_prob = prev_dict[tag]["prob"]
            transition = (tag_given_prevtag[tag][curr_tag])  # tags[tag]
            if curr_tag in word_given_tag[word]:
                emission = (word_given_tag[word][curr_tag])  # tags[curr_tag]
                result = previous_state_prob * transition * emission
            else:
                result = previous_state_prob * transition
            if result > maxi:
                maxi = result
                pointer = tag
    else:
        for tag in prev_dict:         # tags or tgt
            previous_state_prob = prev_dict[tag]["prob"]
            transition = (tag_given_prevtag[tag][curr_tag])          #tags[tag]
            result = previous_state_prob * transition
            if result > maxi:
                maxi = result
                pointer = tag
    return maxi, pointer

def getMaxProbEnd(prev_dict, curr_tag, tag_given_prevtag, word_given_tag):
    maxi = 0
    pointer = ""
    for tag in prev_dict:         # tags or tgt
        previous_state_prob = prev_dict[tag]["prob"]
        transition = (tag_given_prevtag[tag][curr_tag])
        result = previous_state_prob * transition
        if result > maxi:
            maxi = result
            pointer = tag
    return maxi, pointer

def ViterbiAlgorithm(words, tag_given_prevtag, word_given_tag, viterbi):
    #intialization step
    for tag in viterbi[0]:
        transition = (tag_given_prevtag["start_of_line"][tag])
        if words[0] in word_given_tag and tag in word_given_tag[words[0]]:
            emission = (word_given_tag[words[0]][tag])
            viterbi[0][tag]["prob"] = (transition * emission)
        else:
            viterbi[0][tag]["prob"] = transition
        viterbi[0][tag]["backpointer"] = None

    #recursion step
    for i in range(1, len(words)):
        word = words[i]
        for tag in viterbi[i]:
            viterbi[i][tag]["prob"], viterbi[i][tag]["backpointer"] = getMaxProb(viterbi[i-1], word, tag, tag_given_prevtag, word_given_tag)

    #termination step
    end_prob, end_pointer = getMaxProbEnd(viterbi[-1], "end_of_line", tag_given_prevtag, word_given_tag)
    #print(viterbi)
    #sys.exit(0)
    return end_pointer, viterbi

def PerformViterbi(test_data, tag_given_prevtag, word_given_tag):
    tagged_data = []

    for sentence in test_data:
        words = sentence.split(" ")
        viterbi = []
        for i in range(0, len(words)):
            word = words[i]
            viterbi.append(dict())
            if word in word_given_tag:
                for tag in word_given_tag[word]:
                    viterbi[i][tag] = dict()
            else:
                for tag in tag_given_prevtag:
                    viterbi[i][tag] = dict()
        end_pointer, viterbi = ViterbiAlgorithm(words, tag_given_prevtag, word_given_tag, viterbi)

        temp_data = []
        prev_tag = end_pointer
        for i in range(len(words)-1, -1, -1):
            if i == len(words)-1:
                s = words[i] + "/" + prev_tag
            else:
                s = words[i] + "/" + prev_tag + " "
            temp_data.append(s)
            prev_tag = viterbi[i][prev_tag]["backpointer"]
        #print(temp_data)
        #print(viterbi)
        #sys.exit(0)
        tagged_data.append(temp_data)
    return tagged_data

def write_to_file(tagged_data):
    #for l in tagged_data:
    #    l.reverse()
    fh = open("hmmoutput.txt", "w+", encoding="UTF-8")
    for l in tagged_data:
        for pair in l[::-1]:
            fh.write(pair)
        fh.write("\n")
    fh.close()

def compute_accuracy():
    f_original = open("en_dev_tagged.txt", "r", encoding='UTF-8')  # consider encoding UTF-8
    t_original = f_original.read().splitlines()

    f_tagged = open("hmmoutput.txt", "r", encoding='UTF-8')
    t_tagged = f_tagged.read().splitlines()

    count_same = 0
    count_all = 0

    for i in range(len(t_original)):
        word_tag_orig = t_original[i].split(" ")
        word_tag_pred = t_tagged[i].split(" ")
        # flag = 1
        for j in range(len(word_tag_orig)):
            if word_tag_orig[j] == word_tag_pred[j]:
                count_same += 1
            # else:
            #     flag = 0
            count_all += 1
        # if flag == 1:
        #     print(i)

    accuracy_value = (count_same / count_all) * 100
    print("Total = ", count_all)
    print("Correct  = ", count_same)
    print("Accuracy = ", accuracy_value)

def main():
    tag_given_prevtag, word_given_tag = read_file()
    test_data = read_test_file()
    tagged_data = PerformViterbi(test_data, tag_given_prevtag, word_given_tag)
    write_to_file(tagged_data)
    compute_accuracy()

#main()

if __name__ == '__main__':              # to be removed before submission
    hmmlearn3.main()
    t = time.time()
    main()
    print("time taken = ", time.time() - t)

#forget this previous code
'''#def read_file():
tagged_data = []
tags = dict()
word_given_tag = dict()
test_data = []

def getMaxProb(prev_dict, word, curr_tag):
    global tag_given_prevtag, word_given_tag
    maxi = 0
    pointer = ""
    for tag in tag_given_prevtag:         # tags or tgt
        previous_state_prob = prev_dict[tag]["prob"]
        transition = (tag_given_prevtag[tag][curr_tag])          #tags[tag]
        if word in word_given_tag and curr_tag in word_given_tag[word]:
            emission = (word_given_tag[word][curr_tag])                 #tags[curr_tag]
            result = previous_state_prob * transition * emission
        else:
            result = previous_state_prob * transition
        if result > maxi:
            maxi = result
            pointer = tag
    return maxi, pointer

def getMaxProbEnd(prev_dict, curr_tag):
    global tag_given_prevtag
    maxi = 0
    pointer = ""
    for tag in tag_given_prevtag:         # tags or tgt
        previous_state_prob = prev_dict[tag]["prob"]
        transition = (tag_given_prevtag[tag][curr_tag])
        result = previous_state_prob * transition
        if result > maxi:
            maxi = result
            pointer = tag
    return maxi, pointer

def ViterbiAlgorithm(words, viterbi):
    global tags, tag_given_prevtag, word_given_tag
    #intialization step
    for tag in tag_given_prevtag:           # tags or tgt
        transition = (tag_given_prevtag["start_of_line"][tag])
        if words[0] in word_given_tag and tag in word_given_tag[words[0]]:
            emission = (word_given_tag[words[0]][tag])
            viterbi[words[0]][tag]["prob"] = transition * emission
        else:
            viterbi[words[0]][tag]["prob"] = transition
        viterbi[words[0]][tag]["backpointer"] = None

    #recursion step
    for i in range(1, len(words)):
        word = words[i]
        for tag in tag_given_prevtag:        # tags or tgt
            viterbi[word][tag]["prob"], viterbi[word][tag]["backpointer"] = getMaxProb(viterbi[words[i-1]], word, tag, tags, tag_given_prevtag, word_given_tag)

    #termination step
    end_prob, end_pointer = getMaxProbEnd(viterbi[words[-1]], "end_of_line", tags, tag_given_prevtag, word_given_tag)
    return end_pointer, viterbi

def PerformViterbi(tag_given_prevtag):
    global tags, word_given_tag, test_data, tagged_data
    for sentence in test_data:
        words = sentence.split(" ")
        viterbi = dict()
        for word in words:
            viterbi[word] = dict()
            for tag in tag_given_prevtag:
                viterbi[word][tag] = dict()
        end_pointer, viterbi = ViterbiAlgorithm(words, viterbi)
        #print(viterbi)
        #sys.exit(0)

        temp_data = []
        prev_tag = end_pointer
        for i in range(len(words)-1, -1, -1):
            if i == len(words)-1:
                s = words[i] + "/" + prev_tag
            else:
                s = words[i] + "/" + prev_tag + " "
            temp_data.append(s)
            prev_tag = viterbi[words[i]][prev_tag]["backpointer"]

        tagged_data.append(temp_data)

def write_to_file():
    global tagged_data
    print(tagged_data)
    fw = open("sample.txt", "w+", encoding="UTF-8")
    for l in tagged_data:
        for pair in l[::-1]:
            fw.write(pair)
        fw.write("\n")

def compute_accuracy():
    f_original = open("en_dev_tagged.txt", "r", encoding='UTF-8')  # consider encoding UTF-8
    t_original = f_original.read().splitlines()

    f_tagged = open("sample.txt", "r", encoding='UTF-8')
    t_tagged = f_tagged.read().splitlines()

    count_same = 0
    count_all = 0

    for i in range(len(t_original)):
        word_tag_orig = t_original[i].split(" ")
        word_tag_pred = t_tagged[i].split(" ")

        for j in range(len(word_tag_orig)):
            if word_tag_orig[j] == word_tag_pred[j]:
                count_same += 1
            count_all += 1

    accuracy_value = (count_same / count_all) * 100
    print("Accuracy = ", accuracy_value)

def main():
    tag_given_pervtag = dict()
    global tags, word_given_tag, tagged_data, test_data
    f = open("hmmmodel.txt", "r", encoding='UTF-8')  # consider encoding UTF-8
    data = json.load(f)
    tags = data[0]  # count of tags stored in the 3rd line in hmm model
    tag_given_pervtag = data[1]  # count of tag-->nextTag stored in the 7th line in hmm model
    word_given_tag = data[2]  # count of word/tag stored in the 11th line in the hmm model
    prev_total_sentences = data[3]  # count of total number of sentences, will be used in initialization step of viterbi
    # return tags, tag_given_pervtag, word_given_tag, prev_total_sentences
    # def read_test_file():
    fh = open("en_dev_raw.txt", "r", encoding='UTF-8')  # consider encoding UTF-8
    t = fh.read()
    test_data = t.splitlines()
    #print(test_data, type(test_data))
    # return test_data

    #read_file()
    #read_test_file()
    PerformViterbi(tag_given_pervtag)
    write_to_file()
    #compute_accuracy()'''

#previous code for perform viterbi
'''def PerformViterbi(test_data, tags, tag_given_prevtag, word_given_tag, prev_total_sentences):
    tagged_data = []

    for sentence in test_data:
        words = sentence.split(" ")
        viterbi = dict()
        for i in range(0, len(words)):
            word = words[i]
            viterbi[word] = dict()
            for tag in tag_given_prevtag:
                viterbi[word][tag] = dict()
        end_pointer, viterbi = ViterbiAlgorithm(words, tags, tag_given_prevtag, word_given_tag, viterbi, prev_total_sentences)

        temp_data = []
        print(end_pointer)
        print(viterbi)

        prev_tag = end_pointer
        for i in range(len(words)-1, -1, -1):
            if i == len(words)-1:
                s = words[i] + "/" + prev_tag
            else:
                s = words[i] + "/" + prev_tag + " "
            temp_data.append(s)
            prev_tag = viterbi[words[i]][prev_tag]["backpointer"]
        tagged_data.append(temp_data)
        print(temp_data)
        sys.exit(0)
    return tagged_data'''

# Previous code for Viterbi ALgorithm
'''def ViterbiAlgorithm(words, tags, tag_given_prevtag, word_given_tag, viterbi, prev_total_sentences):
    #intialization step
    for tag in tag_given_prevtag:           # tags or tgt
        transition = (tag_given_prevtag["start_of_line"][tag])
        if words[0] in word_given_tag and tag in word_given_tag[words[0]]:
            emission = (word_given_tag[words[0]][tag])
            viterbi[words[0]][tag]["prob"] = (transition * emission)
        else:
            viterbi[words[0]][tag]["prob"] = transition
        viterbi[words[0]][tag]["backpointer"] = None

    #recursion step
    for i in range(1, len(words)):
        word = words[i]
        for tag in tag_given_prevtag:        # tags or tgt          # multiple occurrences of same word.........
            viterbi[word][tag]["prob"], viterbi[word][tag]["backpointer"] = getMaxProb(viterbi[words[i-1]], word, tag, tags, tag_given_prevtag, word_given_tag)

    #termination step
    end_prob, end_pointer = getMaxProbEnd(viterbi[words[-1]], "end_of_line", tags, tag_given_prevtag, word_given_tag)
    return end_pointer, viterbi'''

#previous code for getMaxProb and getMaxProbEnd
'''def getMaxProb(prev_dict, word, curr_tag, tags, tag_given_prevtag, word_given_tag):
    maxi = 0
    pointer = ""
    for tag in tag_given_prevtag:         # tags or tgt
        previous_state_prob = prev_dict[tag]["prob"]
        transition = (tag_given_prevtag[tag][curr_tag])          #tags[tag]
        if word in word_given_tag and curr_tag in word_given_tag[word]:
            emission = (word_given_tag[word][curr_tag])                 #tags[curr_tag]
            result = previous_state_prob * transition * emission
        else:
            result = previous_state_prob * transition
        if result > maxi:
            maxi = result
            pointer = tag
    return maxi, pointer     

def getMaxProbEnd(prev_dict, curr_tag, tags, tag_given_prevtag, word_given_tag):
    maxi = 0
    pointer = ""
    for tag in tag_given_prevtag:         # tags or tgt
        previous_state_prob = prev_dict[tag]["prob"]
        transition = (tag_given_prevtag[tag][curr_tag])
        result = (previous_state_prob * transition)
        if result > maxi:
            maxi = result
            pointer = tag
    return maxi, pointer'''