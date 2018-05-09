<h1> Parts of Speech Tagger </h1>

This is a generalized Parts of Speech (POS) tagger implemented in Python.<br>
The model was trained on the principles of Hidden Markov Model (HMM). <br>
The decoding of data was done using Viterbi Algorithm.<br>
Smoothing was carried out to eliminate the zeros from the transition probabilities(P(word,tag)).<br>
The model trained and tested on data sets of 3 languages (English, Hindi, Chinese). <br>

The results were as follows:<br>

Accuracy for English Data = 88.87%<br>
Accuracy for Hindi Data = 92.41%<br>
Accuracy for Chinese Data = 86.95%<br>

<p>
Implementation details,<br>
1. <b>hmmlearn3.py :</b> includes the code used to train the model using labeled training data set (<b><i>language_name</i>_Train-Dev-Tagged.txt</b>)<br> 
2. The learned transition and emission probabilities computed are stored in <b>hmmmodel.txt</b>. <br>
3. <b>hmmdecode3.py :</b> includes the code for tagging the words in the test data(<b><i>language_name</i>_Test_Raw.txt</b>) using Viterbi Algorithm. and computing the accuracy for it using the transition and emission probabilities passed by the hmmlearn3.py <br>
4. The tagged data is stored in <b>hmmoutput.txt</b>.<br>
5. The predicted tags of our model is compared with the actual results(<b><i>language_name</i>_Test_Tagged.txt</b>) to compute the accuracy. <br>
</p>
