# 💬 Arabic Emotion Detector

### NLP_project_arabic — Arabic Text Emotion Classification

**Understanding emotions in Arabic text through Natural Language Processing.**

NLP_project_arabic is a Natural Language Processing (NLP) project that classifies Arabic text into different emotional categories using machine learning.

The project explores how text preprocessing and numerical text representations can help machine learning models identify emotional patterns in Arabic language data. It also provides an interactive application for users to enter Arabic text and obtain a predicted emotion.

## ✨ Features

* **Arabic Text Classification** — Predicts the emotional category of Arabic text.
* **Five Emotion Categories** — Supports sarcasm, anger, stress, frustration, and neutral text.
* **Text Vectorization** — Converts Arabic text into numerical features using TF-IDF.
* **Machine Learning** — Uses Logistic Regression for multi-class classification.
* **Interactive Web Application** — Provides a simple interface built with Streamlit.
* **Lightweight Model** — Uses traditional machine learning techniques that can run without the computational requirements of large transformer models.

## 🧠 Emotion Categories

| Arabic Label | English Label |
| ------------ | ------------- |
| سخرية        | Sarcasm       |
| غضب          | Anger         |
| توتر         | Stress        |
| إحباط        | Frustration   |
| عادي         | Neutral       |

## ⚙️ How It Works

The project follows a traditional NLP classification pipeline:

1. **Text Input** — The user enters an Arabic sentence or text.
2. **Text Preprocessing** — The input is prepared for feature extraction according to the preprocessing implemented in the project.
3. **TF-IDF Vectorization** — The text is converted into numerical features using Term Frequency–Inverse Document Frequency.
4. **Emotion Classification** — A trained Logistic Regression model predicts the most likely emotion category.
5. **Prediction Display** — The Streamlit interface displays the predicted category.

## 🛠️ Tech Stack

| Technology          | Purpose                             |
| ------------------- | ----------------------------------- |
| Python              | Core implementation                 |
| Scikit-learn        | Machine learning and classification |
| TF-IDF              | Arabic text feature extraction      |
| Logistic Regression | Multi-class emotion classification  |
| Streamlit           | Interactive web application         |
| Pandas              | Dataset handling and analysis       |

## 📊 Model Configuration

The project uses a lightweight text classification pipeline with the following configuration:

* **Feature Extraction:** TF-IDF
* **Maximum Features:** 5,000
* **Classification Algorithm:** Logistic Regression
* **Regularization Parameter:** C = 10
* **Maximum Iterations:** 1,000
* **Number of Classes:** 5

These settings are intended to provide a practical baseline for Arabic emotion classification without requiring GPU acceleration.

## 🚀 Getting Started

### Prerequisites

* Python 3.10 or a compatible Python version.
* Git.

### 1. Clone the Repository

```bash
git clone https://github.com/maram-elaian/NLP_project_arabic.git
cd NLP_project_arabic
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

If the repository includes a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the required packages:

```bash
pip install streamlit scikit-learn pandas
```

### 4. Run the Application

If `main.py` is the Streamlit entry point:

```bash
streamlit run main.py
```

If the application uses a different entry-point filename, replace `main.py` with the appropriate file.

## 🎯 Project Objectives

* Apply Natural Language Processing techniques to Arabic text.
* Explore machine learning approaches to emotion classification.
* Understand the role of TF-IDF in representing textual data.
* Build an end-to-end text classification application.
* Demonstrate a lightweight alternative to computationally expensive language models.

## 🔮 Future Improvements

* Expand and diversify the Arabic dataset.
* Evaluate performance across different Arabic dialects.
* Investigate class imbalance and its effect on predictions.
* Compare Logistic Regression with other machine learning algorithms.
* Explore transformer-based Arabic language models.
* Add per-class precision, recall, F1-score, and a confusion matrix for more comprehensive evaluation.

## 👩‍💻 Team

**Maram Ashraf**

**Haton Jehad**

Artificial Intelligence Student | Machine Learning & NLP

* GitHub: [@maram-elaian](https://github.com/maram-elaian)
* Repository: [NLP_project_arabic](https://github.com/maram-elaian/NLP_project_arabic)

---

*Turning Arabic text into meaningful insights, one sentence at a time.*

