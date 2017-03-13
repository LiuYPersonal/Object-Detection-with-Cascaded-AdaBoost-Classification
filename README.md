# Object-Detection-with-Cascaded-AdaBoost-Classification

<br><b>This is another Course Project.</br></b>

In this project, we will use Viola and Jones approach, a cascaded implementation of the AdaBoost classifier, to design a car detector with arbitrarily low false positive rate.

<br><b>A brief introduction about Cascaded AdaBoost Classifier </b>:</br>
This algorithm is processed in several stages. For each stage, we save the weak classifiers that are constructed using AdaBoost classifier, and only the positive results (both true positive and false positive) detected by classifier will be forwarded to the next stage.
