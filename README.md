# Ctrl+Safe
Ctrl + Safe: Automated Detection of Hate Speech and Cyberbullying in Social Media Through Machine Learning
<br>

<b>Disclaimer: This tool is designed to detect hate speech and offensive content in both Tagalog and English, including instances of cyberbullying. It is intended for informational and educational purposes only and is not guaranteed to be 100% accurate.
Predictions are based on patterns learned by the underlying machine learning model and may vary depending on the quality and context of the input text.
Some sentences may not be accurately classified due to nuances in language, context, slang, sarcasm, or mixed meanings. Users are advised to exercise their own judgment when interpreting the results.</b>

<i>We also included in this GitHub repository the copy of the Jupyter Notebook of the project. You can download the Jupyter Notebook copy, upload it to Google Colab, and run it.</i>

## Created By
Amazing Grace O. Cabiles - Front End Developer <br>
Cyrelle Kristin P. Gapit - Machine Learning Developer <br>
Francen P. Manalo - QA Tester <br>

## Screenshots
<img width="1892" height="909" alt="Image" src="https://github.com/user-attachments/assets/64ecab1f-932f-4cdc-865a-0fa5bd3a2f36" />
<img width="1890" height="905" alt="Image" src="https://github.com/user-attachments/assets/7ef94ec4-1d68-4323-97bc-2fc8ee8d5e72" />
<img width="1891" height="905" alt="Image" src="https://github.com/user-attachments/assets/52cf797b-cf93-4c82-8d24-046c3b6ebe88" />
<img width="1908" height="905" alt="Image" src="https://github.com/user-attachments/assets/66dd8534-5969-4b4b-9aa2-9d39ac39d722" />

## Model Used
Voting Logistic Regression + SVM (Support Vector Machine)

## Note
* Text-only web app (Python Flask)
* Only has <b>Two-class classification (Neutral vs. Hate/Offensive) </b> which may oversimplify sarcasm, indirect insults, or context-dependent cyberbullying texts.
* Performance depends on dataset quality and balance.
* <b>Model cannot detect one word accurately</b> (e.g "Muslim" is flagged as offensive).
* <b>Detected Language </b> feature may inaccurately detect text as Filipino or English.
* No support advance moderation features (filtering, reporting, blocking); It only displays classification results.

## Tech Stack
Flask version 3.1.2 <br>
Python version 3.13.7 <br>
Tailwind CSS version 4.1 <br>
HTML and CSS <br>

## Web App Setup
To run the application install these packages:
```
pip install flask
pip install nltk
pip install langdetect
```

To run the website, type this in the terminal: <br>
`flask run`

## Data Source Links
[Tagalog Dataset from HuggingFace](https://huggingface.co/datasets/syke9p3/multilabel-tagalog-hate-speech)
<br>
[Tagalog Profanity Dataset from HuggingFace](https://huggingface.co/datasets/mginoben/tagalog-profanity-dataset/viewer?views%5B%5D=train)
<br>
[English Dataset from Kaggle](https://www.kaggle.com/datasets/thedevastator/hate-speech-and-offensive-language-detection)
