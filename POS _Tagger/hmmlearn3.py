import json
import time
import math
import sys

def read_file():
    tags = dict()
    word_given_tag = dict()
    tag_given_prevtag = dict()

    f = open("_name_of_training_file_.txt", "r", encoding='UTF-8')       #consider encoding UTF-8
    t = f.read()
    data = t.splitlines()
    total_sentences = len(data)

    for line in data:
        words = line.split(' ')

        #Transition for start of line to first tag
        tag = "start_of_line"
        pos = words[0].rfind('/')
        next_tag = words[0][pos+1:]
        # For Transition probabilities
        if tag in tag_given_prevtag:
            if next_tag in tag_given_prevtag[tag]:
                tag_given_prevtag[tag][next_tag] += 1
            else:
                tag_given_prevtag[tag][next_tag] = 1
        else:
            tag_given_prevtag[tag] = dict()
            tag_given_prevtag[tag][next_tag] = 1

        # Making a tag dictionary for the first tag (start_of_line)
        if tag in tags:
            tags[tag] += 1
        else:
            tags[tag] = 1

        #computing all the probabilities
        for i in range(0, len(words) - 1):
            pair = words[i]
            pos = pair.rfind('/')
            word = pair[0:pos]
            tag = pair[pos+1:]

            next_pair = words[i+1]
            next_pos = next_pair.rfind('/')
            next_tag = next_pair[next_pos+1:]

            #For Transition probabilities
            if tag in tag_given_prevtag:
                if next_tag in tag_given_prevtag[tag]:
                    tag_given_prevtag[tag][next_tag] += 1
                else:
                    tag_given_prevtag[tag][next_tag] = 1
            else:
                tag_given_prevtag[tag] = dict()
                tag_given_prevtag[tag][next_tag] = 1

            #For Emission probabilities
            if word in word_given_tag:
                if tag in word_given_tag[word]:
                    word_given_tag[word][tag] += 1
                else:
                    word_given_tag[word][tag] = 1
            else:
                word_given_tag[word] = dict()
                word_given_tag[word][tag] = 1

            #Making a tag dictionary
            if tag in tags:
                tags[tag] = tags[tag] + 1
            else:
                tags[tag] = 1

        # Transition for last tag to end of line
        pos = words[-1].rfind('/')
        tag = words[-1][pos+1:]
        word = words[-1][0:pos]
        next_tag = "end_of_line"
        # For Transition probabilities of last to end of line
        if tag in tag_given_prevtag:
            if next_tag in tag_given_prevtag[tag]:
                tag_given_prevtag[tag][next_tag] += 1
            else:
                tag_given_prevtag[tag][next_tag] = 1
        else:
            tag_given_prevtag[tag] = dict()
            tag_given_prevtag[tag][next_tag] = 1

        # For Emission probabilities of last tag and word
        if word in word_given_tag:
            if tag in word_given_tag[word]:
                word_given_tag[word][tag] += 1
            else:
                word_given_tag[word][tag] = 1
        else:
            word_given_tag[word] = dict()
            word_given_tag[word][tag] = 1

        # Making a tag dictionary to add last tag           
        if tag in tags:
            tags[tag] += 1
        else:
            tags[tag] = 1

    return tags, word_given_tag, tag_given_prevtag, total_sentences

def write_to_file(tags, word_given_tag, tag_given_prevtag, total_sentences):
    fh = open("hmmmodel.txt", "w+", encoding="UTF-8")
    for prev_tag in tag_given_prevtag:
        for tag in tag_given_prevtag[prev_tag]:
            tag_given_prevtag[prev_tag][tag] = 1.0 * (tag_given_prevtag[prev_tag][tag]) / (tags[prev_tag])
    fh.write(json.dumps(tag_given_prevtag))
    fh.write("\n")
    for word in word_given_tag:
        for tag in word_given_tag[word]:
            word_given_tag[word][tag] = 1.0 * (word_given_tag[word][tag]) / (tags[tag])
    fh.write(json.dumps(word_given_tag))
    fh.write("\n")
    fh.write(json.dumps(tags))
    fh.close()

def smoothing(tags, word_given_tag, tag_given_prevtag):
    #size_tgt = len(tag_given_prevtag)
    n = 2.0
    for main_tag in tag_given_prevtag:
        tags[main_tag] += (len(tag_given_prevtag) * n)
        for tag in tag_given_prevtag:
            if tag in tag_given_prevtag[main_tag]:
                tag_given_prevtag[main_tag][tag] += n
            else:
                tag_given_prevtag[main_tag][tag] = n

    for tag in tag_given_prevtag:
        if "end_of_line" not in tag_given_prevtag[tag]:
            tag_given_prevtag[tag]["end_of_line"] = n

    return tags, word_given_tag, tag_given_prevtag

def main():

    tags, word_given_tag, tag_given_prevtag, total_semtences = read_file()
    tags, word_given_tag, tag_given_prevtag = smoothing(tags, word_given_tag, tag_given_prevtag)
    write_to_file(tags, word_given_tag, tag_given_prevtag, total_semtences)

t = time.time()
main()
print("Train_time = ", time.time() - t)
