# 📧 Email Spam Detection Using Machine Learning

## 📌 Project Overview
This project implements an **AI-based Email Spam Detection system** using **Machine Learning and Natural Language Processing (NLP)**.  
The system classifies emails as **Spam** or **Ham (Not Spam)** and provides a **Flask-based web application** with a login interface for user interaction.

---

## 🎯 Project Objectives
- Detect spam emails using machine learning classification
- Apply NLP techniques for text preprocessing
- Develop a Flask web application with a professional UI
- Provide real-time spam prediction for user-entered email text

---

## 🧠 AI Model & Techniques
- **AI Type:** Supervised Machine Learning  
- **Problem Type:** Binary Classification  
- **Domain:** Natural Language Processing (NLP)  
- **Algorithm Used:** Multinomial Naive Bayes  
- **Feature Extraction:** CountVectorizer (Bi-grams)

---

## 🗂 Dataset
- Labeled email dataset (`emails.csv`)
- Two classes:
  - `0` → Ham (Not Spam)
  - `1` → Spam
- Duplicate records removed during preprocessing

---

## ⚙️ System Workflow
1. User logs in through the web interface  
2. Email text is entered by the user  
3. Text preprocessing and vectorization are applied  
4. Trained ML model predicts Spam or Ham  
5. Result is displayed on the web page  

---

## 🌐 Web Application
- **Backend:** Flask (Python)
- **Frontend:** HTML, CSS
- **Model Storage:** Pickle (`model.pkl`, `vectorizer.pkl`)
- **Features:**
  - Login Page
  - Email Input Form
  - Real-time Spam Detection Result
  - Simple and professional UI

---

## 🚀 Deployment
- Application runs locally using Flask
- Model and vectorizer are loaded at runtime
- paste the index.html file inside an template folder
- Then run the app.py file.
- Run python app.py
---

## 📁 Project Structure
