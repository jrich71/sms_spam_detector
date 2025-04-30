# 📩 SMS Spam Classifier

This project builds an SMS spam detection model using a machine learning pipeline. It uses **TF-IDF vectorization** and a **Linear Support Vector Classifier (LinearSVC)** to predict whether a given SMS text is spam ("spam") or legitimate ("ham"). The project also includes a **Gradio** web application for live interaction with the model.

---

## 👢 Project Structure

- **Data Loading**: Loads SMS messages and their labels from a CSV file.
- **Model Training**: Uses a `Pipeline` with TF-IDF vectorization and a LinearSVC model.
- **Prediction**: Classifies incoming text messages as "spam" or "not spam."
- **Web Interface**: Built with Gradio for easy user interaction.

---

## ⚙️ Requirements

- Python 3.7+
- `pandas`
- `scikit-learn`
- `gradio`

Install required packages using:

```bash
pip install pandas scikit-learn gradio
```

---

## 🚀 How It Works

### 1. Load Dataset

The script loads SMS data from a file `Resources/SMSSpamCollection.csv`. The dataset must have two columns:

- `text_message`: the SMS text
- `label`: "spam" or "ham" (not spam)

```python
sms_text_df = pd.read_csv('Resources/SMSSpamCollection.csv')
```

---

### 2. Train the Model

The `sms_classification` function:

- Splits the dataset into training and testing sets (67% / 33% split).
- Creates a pipeline consisting of:
  - **TfidfVectorizer**: Converts SMS text to numeric features.
  - **LinearSVC**: Learns to classify text as "spam" or "ham."
- Fits the model to the training data.

```python
text_clf = sms_classification(sms_text_df)
```

---

### 3. Make Predictions

The `sms_prediction` function:

- Takes a new text message.
- Uses the trained `text_clf` model to predict whether it is "spam" or "not spam."
- Returns a user-friendly message.

Example:

```python
sms_prediction('you have won $5000! call this number for your prize money!')
```

**Sample Output:**

```
The text message: "you have won $5000! call this number for your prize money!" is spam.
```

---

### 4. Launch the Gradio App

A simple Gradio interface allows users to:

- Enter a text message.
- Get an instant spam/ham prediction.

```python
demo_02 = gr.Interface(
    fn=sms_prediction,
    inputs=gr.Textbox(lines=5, placeholder='[Enter Text Message]', label="Please enter your text or email message here:"),
    outputs=gr.Textbox(lines=5, label="Our model predicted:"),
    title='Spam Checker'
)
demo_02.launch(share=True)
```

The app will open a public link for easy sharing.

---

## 📝 Example Test Messages

Try pasting these into the Gradio app:

1. "You are a lucky winner of $5000!"
2. "You won 2 free tickets to the Super Bowl."
3. "You won 2 free tickets to the Super Bowl text us to claim your prize."
4. "Thanks for registering. Text 4343 to receive free updates on Medicare."

---

## 📌 Notes

- A placeholder `image_classifier` function was included for Gradio testing but is **not** used in the final SMS classification app.
- Future improvements:
  - Enhance preprocessing (e.g., stemming, lemmatization).
  - Experiment with different models (e.g., Logistic Regression, Random Forest).
  - Add model evaluation metrics (e.g., accuracy, precision, recall, F1-score).

