import tkinter as tk
from tkinter import ttk, messagebox
import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import joblib
import torch
import torch.nn as nn
import json
import advertools as adv
from PIL import Image, ImageTk
from transformers import RobertaTokenizer
import emoji
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QFrame
from PyQt5.QtGui import QFont
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

emotions = ['happiness',
            'fear',
            'anger',
            'sadness',
            'disgust',
            'shame',
            'guilt',
            'surprise'
            ]

emoji_keywords = {
    'happiness': ['smile', 'laugh'],
    'fear': ['fear'],
    'anger': ['agry'],
    'sadness': ['sad', 'cry', 'disappointed', 'anxious'],
    'disgust': ['vomiting', 'nauseated face'],
    'shame': ['anxious'],
    'guilt': ['anxious', ],
    'surprise': ['scream']
}


# treba za RoBERTu
class EmotionClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(EmotionClassifier, self).__init__()
        self.bert_model = bert_model
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(bert_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

    def predict(self, input):
        self.eval()  # Set the model to evaluation mode

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        inputs = self.tokenizer(input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.forward(inputs["input_ids"], inputs["attention_mask"])
            _, predicted = torch.max(logits, dim=1)
        return [emotions[predicted.item()]]


def cleanText(text):
    text = text.lower()

    # Remove links
    text = re.sub(r"http\S*|\S*\.com\S*|\S*www\S*", " ", text)

    # Remove @mentions
    text = re.sub(r"\s@\S+", " ", text)

    # Remove all punctuation
    punctuation_table = str.maketrans("", "", string.punctuation)
    text = text.translate(punctuation_table)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    tokens = [token for token in tokens if len(token) > 2]

    return tokens


def open_model_page(name_of_model):
    model_info = data['models'].get(name_of_model, {})

    model_info_window = ModelInfoWindow(name_of_model, model_info)


class ResultWindow(QWidget):
    def __init__(self, input_texts, predicted_emotions, suggested_emojis, name_of_model):
        super().__init__()

        layout = QVBoxLayout()

        # Add a label for the full text
        full_text_label = QLabel(f"Full Text:\n{' '.join(input_texts)}")
        full_text_label.setFont(QFont("Verdana", 12))
        layout.addWidget(full_text_label)

        # Add a separator
        separator_full_text = QFrame()
        separator_full_text.setFrameShape(QFrame.HLine)
        separator_full_text.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator_full_text)

        for i, input_text in enumerate(input_texts):
            result_message = (
                f"Input Text:\n{input_text}\n\nPredicted Emotion: {predicted_emotions[i]}\n\nSuggested Emojis:"
            )

            label = QLabel(result_message)
            label.setFont(QFont("Verdana", 12))
            layout.addWidget(label)

            emoji_message = (
                f" {' '.join(suggested_emojis[i])}"
            )

            emoji_label = QLabel(emoji_message)
            emoji_label.setFont(QFont("Verdana", 20))
            layout.addWidget(emoji_label)

            # Add a separator between individual results
            if i < len(input_texts) - 1:
                separator_result = QFrame()
                separator_result.setFrameShape(QFrame.HLine)
                separator_result.setFrameShadow(QFrame.Sunken)
                layout.addWidget(separator_result)

        layout.setContentsMargins(20, 20, 20, 20)

        self.setLayout(layout)
        self.setWindowTitle(f"Classification Result for {name_of_model}")
        self.setGeometry(100, 100, 600, 400)


def show_result2(input_text, predicted_emotions, suggested_emojis, name_of_model):
    app = QApplication(sys.argv)
    window = ResultWindow(input_text, predicted_emotions, suggested_emojis, name_of_model)
    window.show()
    app.exec_()


class ModelInfoWindow:
    def __init__(self, name_of_model, model_info):
        self.model = models[name_of_model]

        self.root = tk.Toplevel()
        self.root.title(name_of_model)

        description = model_info.get('info', '')
        accuracy = model_info.get('accuracy', '')
        image_filename = model_info.get('confusion_matrix_file_name', '')

        window_width = 880
        window_height = 450
        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        self.root.minsize(window_width, window_height)
        self.root.maxsize(window_width, window_height)

        title_label = ttk.Label(self.root, text=name_of_model, font=("Verdana", 20))
        title_label.grid(row=0, column=0, columnspan=7, pady=20, sticky='n')

        description_label = ttk.Label(self.root, text=f"{description}", font=("Verdana", 10), wraplength=400, anchor='w')
        description_label.grid(row=1, column=0, columnspan=7, pady=10, padx=10, sticky='w')

        accuracy_label = ttk.Label(self.root, text=f"Accuracy: {accuracy}", font=("Verdana", 12))
        accuracy_label.grid(row=2, column=0, columnspan=3, pady=10, padx=10, sticky='w')

        # Load and display the image on the right with a larger size
        if image_filename:
            image_path = f"images/{image_filename}"
            image = Image.open(image_path)
            image.thumbnail((350, 350))
            tk_image = ImageTk.PhotoImage(image)

            image_label = ttk.Label(self.root, image=tk_image)
            image_label.grid(row=1, column=7, rowspan=4, padx=20, pady=10, sticky='e')

            image_label.image = tk_image

        self.text_entry = tk.Text(self.root, width=40, height=5)
        self.text_entry.grid(row=3, column=0, columnspan=3, pady=10, padx=10, sticky='we')

        classify_button = ttk.Button(self.root, text="Classify", command=lambda m=name_of_model: self.show_result(m),
                                     width=20)
        classify_button.grid(row=3, column=3, pady=10, padx=10, sticky='w')



    def show_result(self, name_of_model):
        input_text = self.text_entry.get("1.0", tk.END)
        #predicted_emotion = classify_text(self.model, self.text_entry.get("1.0", tk.END))
        suggested_emojis, predicted_emotions = suggest_emojis(self.model, self.text_entry.get("1.0", tk.END))

        show_result2(sent_tokenize(input_text), predicted_emotions, suggested_emojis, name_of_model)


def classify_text(model, text_to_classify):
    print()
    print("Sentence:", text_to_classify)
    predicted_emotion = model.predict([text_to_classify])
    return predicted_emotion[0]


def suggest_emojis_from_text(text):
    words = cleanText(text)
    suggested_emojis = []
    for word in words:
        suggested_emoji = emoji.emojize(f":{word}:")
        if suggested_emoji != f":{word}:":
            suggested_emojis.append(suggested_emoji)
        suggested_emoji = adv.emoji_search(word)['emoji'].head(3).to_list()
        if suggested_emoji is not None:
            suggested_emojis.extend(suggested_emoji)
    return suggested_emojis


def suggest_emojis(model, text):
    sentences = sent_tokenize(text)
    suggested_emojis_result = []
    predicted_emotions = []
    for sent in sentences:
        temp = []
        predicted_emotion = classify_text(model, sent)
        predicted_emotions.append(predicted_emotion)
        print("Predicted emotion:", predicted_emotion)
        # from emotion keywords
        for suggestion in emoji_keywords[predicted_emotion]:
            search_result = (adv.emoji_search(suggestion)['emoji'].head(3)).to_list()
            temp.extend(search_result)

        # jos dodamo na temelju teksta
        temp.extend(suggest_emojis_from_text(sent))

        suggested_emojis_set = set(temp)
        print("Suggested emojis: ", ' '.join(suggested_emojis_set))
        suggested_emojis_result.append(suggested_emojis_set)
    return suggested_emojis_result, predicted_emotions


# ucitavanje podataka koji su zapisani u datoteku
with open('info_file.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# ucitavanje modela
naive_bayes_model = joblib.load('models/naive_bayes_pipeline.pkl', 'r')
logistic_regression_model = joblib.load('models/logistic_regression_pipeline.pkl', 'r')
svm_model = joblib.load('models/svm_pipeline.pkl', 'r')
#bilstm_model = load_model('models/bilstm_model.keras')
roberta_model = pickle.load(open('models/RoBERTa.pkl', 'rb'))
# bert_model = ktrain.load_predictor("models/bert_model")

models = {'Naive Bayes': naive_bayes_model,
          'Logistic Regression': logistic_regression_model,
          'SVM': svm_model,
          #'BiLSTM': bilstm_model,
          # 'BERT': bert_model,
          'RoBERTa': roberta_model
          }

work_from_console = False

if work_from_console:
    while True:
        user_input = input("Enter your text: ")
        print("Choose a model:")
        for i, model in enumerate(models, start=1):
            print(f"{i}. {model}")

        while True:
            selected_index = int(input("Enter the number of the model you want to choose: "))
            if 1 <= selected_index <= len(models):
                selected_model_name = list(models.keys())[selected_index - 1]
                print(f"You chose: {selected_model_name}")
                selected_model = models[selected_model_name]
                break
            else:
                print("Invalid choice. Please enter a valid number.")

        suggest_emojis(selected_model, user_input)
        print()
else:
    # namjestanje prozora
    root = tk.Tk()
    root.title(data['title'])

    window_width = 800
    window_height = 400

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_position = (screen_width - window_width) // 2
    y_position = (screen_height - window_height) // 2

    min_window_width = window_width
    min_window_height = window_height
    root.minsize(min_window_width, min_window_height)
    root.maxsize(min_window_width, min_window_height)

    root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

    # naslov
    title_label = ttk.Label(root, text=data['title'], font=("Verdana", 20), wraplength=700, anchor='center', justify='center')
    title_label.pack(side=tk.TOP, pady=20)

    # podnaslov
    authors_label = ttk.Label(root, text="Authors: " + data['authors'], font=("Verdana", 12))
    authors_label.pack(side=tk.TOP)

    # opis
    description_label = ttk.Label(root, text=data['description'], font=("Verdana", 10), wraplength=750, anchor='center', justify='center')
    description_label.pack(side=tk.TOP, pady=10)

    # namjestanje gumba
    button_frame = ttk.Frame(root)
    button_frame.pack(side=tk.TOP, pady=10)

    button_width = 20
    button_padding = 5

    for model_name in models:
        model_button = ttk.Button(button_frame, text=model_name, command=lambda m=model_name: open_model_page(m),
                                  width=button_width)
        model_button.pack(side=tk.LEFT, padx=button_padding, pady=button_padding)
    root.mainloop()
