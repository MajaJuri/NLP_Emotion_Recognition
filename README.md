# Emoji Recommender System

This project is a part of the course "Natural Language Processing" at FER. We implemented multiple machine learning, deep learning and transformer
models for an emotion recognition task in order to use those predictions to recommend emojis based on the input text. Additionally, we used a 
keyword-based approach to also recommend emojis that are related to the context of the input text. 

For every model or group of models there is a separate Jupyter Notebook with the code used to preprocess the data, train the models and evaluate them.

- "All Things Data" - a notebook for data preprocessing and a simple EDA
- "Tradititonal ML" - a notebook for the machine learning algorithms (SVM, Naive Bayes, Logistic Regression)
- "BERT i RoBERTa" - a notebook for BERT and RoBERTa models
- "Deep_Learning_Models" - a notebook for the Bi-LSTM model
- "Llama2" - a notebook for the Llama model
- "Phi2" - a notebook for the Phi model

The "emotion_classifier_app" is a Python script for displaying the graphical user interface our emoji recommender system. While not all of our models are presently integrated, users can choose from a lineup featuring Naive Bayes, SVM, Logistic Regression, BERT, and RoBERTa. After selecting a model, a window opens up revealing additional information about the selected model. There's also a text box ready for user input — just type in the text you'd like to receive emoji recommendations for.

Due to GitHub's storage limitations, deployment of the Emoji Recommendation app is currently unavailable. However, we've prepared a video demo (Emoji Recommender app video demo.mp4) to offer you a glimpse of its functionality.

To delve deeper into our research and understand the creation of the Emoji Recommender system, check out our paper, "A Comparative Study of Language Models."
