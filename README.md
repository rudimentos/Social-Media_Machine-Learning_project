# Machine Learning project on python with Sentiment analysis

## This is a diploma project involving cleaning a twitter dataset, building a machine learning model and displaying results dynamically on a prototype webpage. 

### Description
Customer support on social media is an important aspect of PR to manage by companies, as they directly affect the customers and the brand image. A conversation between a customer and the support may escalate or deescalate based on a myriad of reasons. This project explores linguistic and sentiment factors, as well as timing of the response etc.

The dataset consists of tweets on Twitter (now X) between customers and large companies, including Amazon, Apple and Uber. 

It is publicly available on Kaggle through https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
## Completed steps in order:
  1. Importing and cleaning dataset
  2. Reconstructing threads from tweets
  3. Exploratory Data Analysis
  4. Feature engineering (Linguistic, Structural and Sentiment)
  5. Hypothesis testing (Mann-Whitney U, Chi-Squared)
  6. Preprocessing
  7. Building and evaluating the ML model
  8. Exporting it into a website prototype
## Languages and platforms used:
* Python
  * pandas
  * numpy
  * scikit-learn
  * XGBoost
  * spaCy
  * matplotlib
  * SHAP

* VSCode (Jupyter Notebook)

* GitHub version control

* HTML, CSS and JavaScript (almost entirely via ClaudeAI LLM, i'm a Data specialist, not a WebDev)
# Reproducing results

  1. Download the dataset from https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter into the folder of the .ipynb file.
  2. Run the entire .ipynb file. (Note: it will take up to an hour due to spaCy and HuggingFace)
  3. Run the app.py

# Screenshots
## Data cleaning

The dataset had no duplicate or missing values, but it did have orphaned tweets, whose thread root got truncated during data collection. Additionally, I converted some columns' datatypes and removed non-English tweets.

<img width="1037" height="632" alt="image" src="https://github.com/user-attachments/assets/acb01194-8c25-4bea-aa9b-a863f644a9b4" />
<br>
<br>
<img width="386" height="141" alt="image" src="https://github.com/user-attachments/assets/73a96542-cf78-404d-b17e-20feb5127e13" />
<br>
<br>
<img width="782" height="394" alt="image" src="https://github.com/user-attachments/assets/1c548bc0-863c-4733-a6e0-3bde6bfeffee" />

## Exploratory Data Analysis

EDA was done through pandas and matplotlib
<br>
<br>
<img width="589" height="689" alt="image" src="https://github.com/user-attachments/assets/d3ea253f-61f6-493a-b054-5436d5b13d88" />
<br>
<br>
<img width="487" height="384" alt="image" src="https://github.com/user-attachments/assets/e2b0358f-260f-4a73-b50a-d55e16a0e44a" />

## Feature engineering

Matching generic negativity drivers in text

<img width="706" height="629" alt="image" src="https://github.com/user-attachments/assets/f01cf08f-d22a-49bc-ad6d-1f26ce654389" />

<img width="492" height="313" alt="image" src="https://github.com/user-attachments/assets/3cba3a74-d023-49cf-8f8e-d0745e380f2f" />

Here I used RoBERTa twitter sentiment model for Sentiment analysis

<img width="763" height="551" alt="image" src="https://github.com/user-attachments/assets/f98bacb6-38a0-42cc-94c2-5aeaa69402c1" />



## Preprocessing, Model Building and Evaluating
scikit-learn for Scaler and splitting data
<img width="748" height="295" alt="image" src="https://github.com/user-attachments/assets/17727e44-7114-420b-b102-482b6bab75a5" />
<br>
<br>
The actual model was XGBoost
<br>
<img width="600" height="565" alt="image" src="https://github.com/user-attachments/assets/431c88a4-bb18-4334-b98c-8d9aef204f6a" />
<br>
<br>
<img width="335" height="323" alt="image" src="https://github.com/user-attachments/assets/5969df5d-4c72-4692-9327-e1ed2cb0656f" />
<br>
<br>
<img width="531" height="382" alt="image" src="https://github.com/user-attachments/assets/9689552e-e692-4202-b984-914fcc99ed6f" />

## Webpage results
This is the evaluator and predictor of negativity and escalation of threads. It accepts input with option of sample threads from the dataset. The website was done entirely through LLM code.
<img width="940" height="470" alt="image" src="https://github.com/user-attachments/assets/db5ae118-9c8f-4f1c-94c2-c13b608a0426" />

This is a dashboard that shows the results of the ML model on this particular dataset. It is interactive: you can tweak the decision threshold. Hovering over the bars displays information.

<img width="940" height="467" alt="image" src="https://github.com/user-attachments/assets/3241ea32-2bdf-4e2a-8cbd-e6375d7ee32b" />

