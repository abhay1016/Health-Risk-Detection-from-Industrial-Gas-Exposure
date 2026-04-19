# 🛡️ HealthGuard AI - Health Risk Detection from Industrial Gas Exposure

An AI-powered machine learning system that predicts health risks based on industrial gas exposure levels. Built with Python, Scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**HealthGuard AI** is a machine learning application designed to assess health risks for workers exposed to industrial gases. By analyzing concentration levels of various gases and exposure duration, the system predicts potential health conditions including:

- ✅ **Healthy** - No significant risk
- ⚠️ **Asthma** - Moderate respiratory risk
- ⚠️ **Bronchitis** - Moderate respiratory risk
- 🚨 **COPD** - High respiratory risk
- 🚨 **Lung Cancer** - Severe health risk

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **Multi-Model Training** | Trains Logistic Regression, Random Forest, and XGBoost (optional) |
| 📊 **Model Comparison** | Automatically selects the best performing model |
| 🎨 **Modern UI** | Sleek, startup-style dark theme interface |
| ⚡ **Real-Time Prediction** | Instant health risk assessment |
| 📈 **Probability Distribution** | Shows confidence for all possible conditions |
| 💡 **Recommendations** | Provides actionable health advice |
| 🔄 **End-to-End Pipeline** | From data preprocessing to web deployment |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/health-risk-detection.git
cd health-risk-detection
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Step 1: Train the Model

```bash
python main.py
conda activate health 
```

This will:
- Generate a synthetic dataset (1000 samples)
- Preprocess the data
- Train multiple models
- Evaluate and compare models
- Save the best model to `model.pkl`

### Step 2: Launch the Web Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Step 3: Make Predictions

1. Enter gas concentration values:
   - CO (Carbon Monoxide) - ppm
   - NO₂ (Nitrogen Dioxide) - ppb
   - SO₂ (Sulfur Dioxide) - ppb
   - O₃ (Ozone) - ppb
   - Benzene - ppb
   - Toluene - ppb
   - Xylene - ppb
   - Exposure Duration - hours

2. Click **"ANALYZE HEALTH RISK"**

3. View the prediction, confidence score, and recommendations

---

## 📁 Project Structure

```
Health Risk Prediction/
│
├── main.py                 # Model training script
│   ├── Data generation
│   ├── Preprocessing
│   ├── Model training
│   ├── Evaluation
│   └── Model saving
│
├── app.py                  # Streamlit web application
│   ├── Modern dark UI
│   ├── Input forms
│   ├── Prediction display
│   └── Recommendations
│
├── model.pkl               # Saved trained model
├── health_gas_data.csv     # Generated dataset
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ⚙️ How It Works

### 1. Data Preprocessing

```
Raw Data → Handle Missing Values → Label Encoding → Feature Scaling → Train/Test Split
```

- **Missing Values**: Filled with column median
- **Label Encoding**: Converts disease names to numeric labels
- **Feature Scaling**: StandardScaler normalization
- **Split Ratio**: 70% training, 30% testing

### 2. Model Training

Three models are trained and compared:

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear classifier, fast and interpretable |
| **Random Forest** | Ensemble of decision trees, handles non-linearity |
| **XGBoost** | Gradient boosting, high accuracy (optional) |

### 3. Prediction Pipeline

```
User Input → Scale Features → Model Prediction → Decode Label → Display Result
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.33% |
| **Precision** | 99% (macro avg) |
| **Recall** | 99% (macro avg) |
| **F1-Score** | 99% (macro avg) |

### Confusion Matrix

```
              Predicted
            As  Br  CO  He  LC
Actual As  [60   0   0   0   0]
       Br  [ 1  58   1   0   0]
       CO  [ 0   0  60   0   0]
       He  [ 0   0   0  60   0]
       LC  [ 0   0   0   0  60]
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Programming language |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **Scikit-learn** | Machine learning |
| **XGBoost** | Gradient boosting (optional) |
| **Streamlit** | Web application framework |
| **Pickle** | Model serialization |

---

## 🔧 Configuration

### Using Your Own Dataset

1. Prepare a CSV file with the following columns:
   - `CO`, `NO2`, `SO2`, `O3`, `Benzene`, `Toluene`, `Xylene`
   - `Exposure_Duration`
   - `Disease` (target column)

2. Update `main.py`:
   ```python
   # Comment out sample generation
   # df = generate_sample_dataset(...)

   # Load your data
   df = load_data('your_dataset.csv')
   ```

3. Retrain the model:
   ```bash
   python main.py
   ```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

This application is for **educational purposes only**. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

<p align="center">
  Built with  by <strong>HealthGuard AI</strong>
</p>
