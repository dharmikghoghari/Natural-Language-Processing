<h1> Perceptron Model </h1><br>

The model was develoepd on the principles of <b>Perceptorn Algorithm</b>.<br>
Two flavors of the algorithm were implemented, Vanilla Perceptron and Averaged Perceptron.<br>
The model's accuracy was improved by using a list of stop words.<br> 
The model was trained and tested on the Deceptive Opinion Spam Corpus v1.4(licensed under the Creative Commons).<br>

The below results were obtained when model was trained on 1280 reviews and the model was used to classify 320 reviews.<br>

Vanilla Perceptron Model <br>
Accuracy: 87.812%<br>
F1 Score: 87.812% <br>

Averaged Perceptron Model<br>
Accuracy: 88.281%<br>
F1 Score: 88.281%<br>

Run time: 2.2 seconds (includes training on 1280 reviews and classifying 320 reviews for both the vanilla and averaged model) 

Implementation details,<br>

1. <b>perceplearn3.py :</b> includes the code used to train the model using labeled training data set (dev-train-labeled.txt)<br> 
2. The learned weights and bias of the perceptron models are stored seperately in <b>vanillamodel.txt</b> for vanilla model and in <b>averagedmodel.txt</b> for averaged model. <br>
3. <b>percepclassify3.py :</b> includes the code for classifying the test data(<b>test-text.txt</b>) and computing the accuracy and F1 Score for it using the parameters passed by the perceplearn3.py <br>
4. The classification results are stored in <b>percepoutput.txt</b>.<br>
5. The classification results of our model is compared with the actual results(<b>test-key.txt</b>) to compute the accuracy and F1 Score. <br>

The algorithm details can be found here,<br>
Reference: Hal Daumé III, A Course in Machine Learning (v. 0.99 draft), Chapter 4: The Perceptron<br>
Link: <b>http://www.ciml.info/dl/v0_99/ciml-v0_99-ch04.pdf</b>
