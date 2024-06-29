##Results Analysis

The model was trained for 10 epochs on the provided dataset. The training and validation loss, along with accuracy, were recorded at each epoch. Below is the summary of the final evaluation metrics on the validation set:

	•	Accuracy: 0.8462
	•	Precision: 0.8462
	•	Recall: 1.0000
	•	F1 Score: 0.9167

##Confusion Matrix

The confusion matrix is a key tool for understanding the performance of a classification model. It provides insight into how well the model is performing by displaying the true positives, true negatives, false positives, and false negatives.

In our case, the confusion matrix is as follows:

##Predicted
          Real  Fake
True Real   0    2
     Fake   0   11

##Interpretation:

	•	True Real (Real, Real): The model did not predict any real samples correctly (0 true positives).
	•	False Real (Real, Fake): The model incorrectly classified 2 real samples as fake (2 false negatives).
	•	True Fake (Fake, Fake): The model correctly identified 11 fake samples (11 true positives).
	•	False Fake (Fake, Real): There were no fake samples misclassified as real (0 false positives).

##Detailed Metrics Explanation

	•	Accuracy: The overall accuracy of the model is 84.62%. This means that out of all the samples, 84.62% were correctly classified.
	•	Precision: Precision for the “fake” class is 84.62%. This indicates that when the model predicts a sample as fake, 84.62% of the time it is correct.
	•	Recall: Recall for the “fake” class is 100%. This means that the model successfully identified all the fake samples in the validation set.
	•	F1 Score: The F1 score, which is the harmonic mean of precision and recall, is 91.67%. This metric provides a balance between precision and recall.
