# ♻️ AI Waste Classifier

### A Deep Learning System for Smart Waste Sorting

Powered by **TensorFlow**, **EfficientNetB0**, and **Streamlit**

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields/badge/TensorFlow-2.15-orange.svg)
![Streamlit](https://img.shields/badge/Streamlit-1.39-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Overview

The **AI Waste Classifier** is an intelligent waste-sorting system built with deep learning. It classifies waste images into:

* ♻️ **Recyclable**
* 🌱 **Biodegradable**
* 🚯 **Non-Recyclable**

The model is trained using **EfficientNetB0** with transfer learning and fine-tuning, achieving **90%+ accuracy**.
A beautiful **Streamlit web app** allows users to:

* Upload an image
* Or take a picture using their camera
* Get real-time waste classification
* View confidence levels
* Receive proper disposal guidance

---

## ✨ Features

✔ Upload or take a picture
✔ Realtime predictions
✔ 3 waste categories
✔ Confidence level display
✔ AI-powered eco-friendly advice
✔ Beautiful UI with custom styling
✔ Mobile-friendly
✔ Works locally or on Streamlit Cloud

---

## 🧠 Model Overview

* **Architecture:** EfficientNetB0
* **Image Size:** 224 × 224
* **Training:** Transfer learning + fine-tuning
* **Metrics:** Accuracy, Precision, Recall, F1-score
* **Dataset:** Recyclable, Biodegradable, Non-Recyclable

---

## 📂 Project Structure

```
ai-waste-classifier/
│
├── app.py                          # Streamlit frontend
├── waste_model_efficientnet.keras  # Trained model
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/                            # Optional scripts
│   ├── train.py
│   ├── preprocess.py
│   └── utils.py
```

---

## 🛠️ Installation

### 1️⃣ Clone the repo

```bash
git clone https://github.com/bechosen-spec/ai-waste-classifier.git
cd ai-waste-classifier
```

### 2️⃣ Create a virtual environment

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

App opens at:

👉 [http://localhost:8501](http://localhost:8501)

---

## 📦 Datasete

The dataset consists of three folders:

* `recyclable/`
* `biodegradable/`
* `non_recyclable/`

Originally sourced and reorganized from:

* Hazardous (ignored)
* Non-Recyclable
* Organic
* Recyclable

---

## 🧪 Training the Model

Training was done in Google Colab using:

* Dataset cleaning
* Preprocessing & augmentation
* Transfer learning (EfficientNetB0)
* Fine-tuning
* Evaluation (accuracy, precision, recall, F1)
* Saving final `.keras` model

---

## 🔮 Future Enhancements

* Mobile app (React Native / Flutter)
* TensorFlow Lite deployment
* Hardware integration with smart bins
* Enhanced multi-class waste categories
* Real-time object detection

---

## 👨🏾‍💻 Author

**Prosper Nsohlebna Atiah**


---

## 📄 License

This project is licensed under the **MIT License**.

---

## ⭐ Support the Project

If this helped you, please **star the repository on GitHub** — it encourages further development!
#
