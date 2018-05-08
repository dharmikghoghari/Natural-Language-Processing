<h1>Naive Bayes Model</h1><br>

The model was develoepd on the principles of Naive Bayes algorithm. <br>
The model's accuracy was improved by almost 2% using a list of stop words. <br>
The model was trained and tested on the Deceptive Opinion Spam Corpus v1.4(licensed under the Creative Commons).<br>

Accuracy of the model: 90.625%<br>
F1 score of the model: 90.619%<br>
Run time: 1.2 seconds (includes training on 1280 reviews and classifying 320 reviews) <br>

Implementation details,<br>
1. <b>nblearn3.py :</b> includes the code used to train the model using labeled training data set (dev-train-labeled.txt)<br> 
2. Learned parameters of the Naive Bayes model is stored in <b>nbamodel.txt</b>.<br>
3. <b>nbclassify3.py :</b> includes the code for classifying the test data(<b>test-text.txt</b>) and computing the accuracy and F1 Score for it using the parameters passed by the nblearn3.py <br>
4. The classification results are stored in <b>nboutput.txt</b>.<br>
5. The classification results of our model is compared with the actual results(<b>test-key.txt</b>) to compute the accuracy and F1 Score. <br>
