# Object-Detection-with-Cascaded-AdaBoost-Classification

<br><b>This is another Course Project.</b></br>

In this project, we will use Viola and Jones approach, a cascaded implementation of the AdaBoost classifier, to design a car detector with arbitrarily low false positive rate.

<br><b>Extract features file:</b> extract features using haar detector.</br>
<br><b>cascadeAdaBoost file:</b> find weak classifier iteratively, then form a strong classifier.</br>
<br><b>classify file:</b> use the strong classifier to classify images.</br>

<br><b>A brief introduction about Cascaded AdaBoost Classifier </b>:</br>
This algorithm is processed in several stages. For each stage, we save the weak classifiers that are constructed using AdaBoost classifier, and only the positive results (both true positive and false positive) detected by classifier will be forwarded to the next stage.

Since there is a huge amount of calculation for feature extraction and classifier calculation. We save and load the calculated results in each phase instead of passing the values directly.


