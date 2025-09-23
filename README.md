# Intelligent Power System Fault Detection and Monitoring

![GitHub Banner](https://user-images.githubusercontent.com/8123558/150153963-39045b40-722a-45a9-a681-6945a08573b9.png)

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [How It Works: The Project Pipeline](#-how-it-works-the-project-pipeline)
  - [1. Synthetic Data Generation](#1-synthetic-data-generation)
  - [2. Feature Extraction](#2-feature-extraction)
  - [3. Model Training & Evaluation](#3-model-training--evaluation)
  - [4. Real-time Monitoring Dashboard](#4-real-time-monitoring-dashboard)
- [Understanding the Training Results](#-understanding-the-training-results)
  - [Example Waveforms](#example-waveforms)
  - [Model Performance Comparison](#model-performance-comparison)
  - [Feature Importance](#feature-importance)
  - [Confusion Matrix](#confusion-matrix)
- [🚨 Detecting Power Line Cuts During Disasters](#-detecting-power-line-cuts-during-disasters)
- [🚀 How to Run the Project](#-how-to-run-the-project)

## 🌟 Project Overview

This project presents an end-to-end solution for the intelligent detection and classification of electrical faults in a three-phase power system. Leveraging machine learning, the system can automatically identify various anomalous conditions, such as short circuits, overloads, and line cuts, by analyzing current waveform data.

The core of the project is a robust machine learning model trained on a comprehensive synthetic dataset that mimics real-world power system behavior. This is complemented by an interactive web-based dashboard that provides real-time monitoring, fault simulation, and historical analysis capabilities. The goal is to create a tool that can enhance the reliability and safety of power grids, reduce diagnostic time, and enable quicker responses to critical failures.

## ✨ Key Features

*   **Synthetic Data Engine:** Generates realistic three-phase current waveforms for five distinct scenarios: Normal Operation, Short Circuit, Ground Fault, Overload, and LT Line Cut.
*   **Advanced Feature Engineering:** Extracts 24 distinct features from raw waveform data, covering time-domain, frequency-domain, and multi-phase characteristics.
*   **Multi-Model Training Pipeline:** Trains and evaluates four different machine learning classifiers (Random Forest, Gradient Boosting, SVM, Neural Network) to identify the best-performing model.
*   **Interactive Web Dashboard:** A user-friendly interface built with Streamlit for:
    *   **Real-time Monitoring:** Simulates a live power feed and instantly flags detected faults.
    *   **Manual Fault Injection:** Allows users to manually test the model's performance on specific fault types.
    *   **Historical Analysis:** Generates and visualizes historical fault data to identify trends and patterns.

## 💻 Technology Stack

*   **Data Science & Computation:** Python, NumPy, Pandas, SciPy
*   **Machine Learning:** Scikit-learn
*   **Data Visualization:** Matplotlib, Seaborn, Plotly
*   **Web Framework:** Streamlit
*   **Model Persistence:** Joblib

## 🛠️ How It Works: The Project Pipeline

The project follows a systematic four-step pipeline, from data creation to a fully functional monitoring application.

### 1. Synthetic Data Generation

Real-world fault data is scarce, inconsistent, and dangerous to collect. To overcome this, we developed a sophisticated data generator that creates a balanced and diverse dataset. The generator simulates three-phase current waveforms by mathematically modeling the following scenarios:
*   **Normal Operation:** Ideal, balanced sinusoidal waveforms with minor noise.
*   **LT Line Cut:** An open-conductor fault where the current in one phase abruptly drops to zero.
*   **Ground Fault:** A short-duration, high-amplitude surge on a single phase, representing a connection to the ground.
*   **Short Circuit:** A sudden, sustained, and very large increase in current on one or more phases.
*   **Overload:** A gradual and sustained increase in the current amplitude across all phases beyond the nominal rating.

To ensure realism, each generated waveform includes random variations in amplitude, phase, noise levels (Signal-to-Noise Ratio), and harmonic distortion.

### 2. Feature Extraction

Raw waveform data, which consists of thousands of data points per second, is not suitable for direct input into traditional machine learning models. Therefore, we perform feature extraction to convert this complex data into a structured set of meaningful numerical indicators. For each 2-second waveform sample, we calculate 24 features, including:
*   **Time-Domain Features:** RMS (Root Mean Square) current, peak amplitude, standard deviation, and mean. These describe the magnitude and variability of the signal.
*   **Frequency-Domain Features:** Magnitude of the fundamental frequency (50 Hz) and Total Harmonic Distortion (THD). These describe the purity and shape of the sine wave.
*   **Signal Features:** The rate of zero crossings, which can indicate frequency changes or high-frequency noise.
*   **Three-Phase Features:**
    *   **Phase Imbalance:** The difference between the maximum and minimum RMS values among the three phases. This is a critical indicator for many fault types.
    *   **Zero Sequence Current:** The vector sum of the three-phase currents, which should be near zero in a balanced system.

### 3. Model Training & Evaluation

This phase aims to find the most accurate and reliable classification model.
1.  **Data Preparation:** The feature-extracted dataset is split into training and testing sets. The features are scaled to ensure that no single feature disproportionately influences the model.
2.  **Model Benchmarking:** We train four different types of classifiers: Random Forest, Gradient Boosting, Support Vector Machine (SVM), and a Multi-Layer Perceptron (Neural Network).
3.  **Performance Comparison:** Each model's performance is evaluated using **Cross-Validation** on the training set to ensure stability and **Test Accuracy** on the unseen test set to measure its generalization capability.
4.  **Hyperparameter Tuning:** The best-performing model from the benchmark (in this case, Random Forest) is selected and further optimized using `GridSearchCV`. This process systematically searches for the best combination of model parameters to maximize its predictive accuracy.
5.  **Model Serialization:** The final, optimized model, along with the data scaler and label encoder, is saved to disk (`.pkl` files). This allows the application to load and use the trained model without needing to retrain it every time.

### 4. Real-time Monitoring Dashboard

The final step is a user-facing application built with Streamlit. This dashboard loads the saved model and provides a powerful interface for interaction. It operates in three modes:
*   **Real-time Simulation:** Continuously generates new waveform data, injects faults with a predefined probability, and uses the model to predict the system's status in real-time. It visualizes the waveforms and classification probabilities.
*   **Manual Fault Injection:** Allows a user to select a specific fault type, generate a corresponding waveform, and see how the model classifies it. This is excellent for testing and demonstration.
*   **Historical Analysis:** Simulates historical data over a chosen period, identifies past fault events, and visualizes trends through interactive charts and data tables.

## 📊 Understanding the Training Results

The model training process generates several key visualizations that help us understand its performance and behavior.

### Example Waveforms


This plot displays a representative sample of the three-phase current waveforms for each of the five simulated scenarios. It serves as a visual guide to what the model is learning to differentiate.
*   **Normal Waveform:** Shows three perfectly balanced, clean sinusoidal waves, 120 degrees out of phase with each other.
*   **LT_Line_Cut Waveform:** Demonstrates a scenario where one phase (Phase C, in blue) is suddenly cut, and its current drops to zero.
*   **Ground_Fault Waveform:** Illustrates a brief, sharp, and high-magnitude spike on one phase (Phase A, in red), characteristic of a ground fault.
*   **Short_Circuit Waveform:** Shows a dramatic and sustained increase in the current amplitude across all three phases after the fault occurs.
*   **Overload Waveform:** Depicts a gradual increase in the amplitude of all three waveforms over time, indicating a steadily rising load.

### Model Performance Comparison


This bar chart compares the performance of the four candidate models.
*   **Test Accuracy (Blue Bar):** This shows the accuracy on the unseen test data. A high value here indicates that the model generalizes well to new data.
*   **CV Mean (Orange Bar):** This represents the average accuracy score from 5-fold cross-validation. The small error bars indicate that the model's performance is stable and consistent across different subsets of the data.

**Conclusion:** All four models performed exceptionally well on this synthetic dataset. The **Random Forest** model was chosen as the best because it achieved perfect test accuracy (1.0) and a very high, stable cross-validation score (1.0), making it both highly accurate and reliable.

### Feature Importance


This plot reveals which features the final Random Forest model considered most important when making its classification decisions.
*   The most influential feature by a significant margin is **`Phase_Imbalance`**. This makes perfect sense, as faults like line cuts, ground faults, and single-phase short circuits directly cause an imbalance in the currents of the three phases.
*   Other important features include **Total Harmonic Distortion (`THD`)** of the phases, the **`Avg_RMS`** current, and the **standard deviation (`Std`)** of the phases. This tells us that the model relies heavily on changes in waveform shape, overall current magnitude, and signal stability to detect anomalies.

### Confusion Matrix


A confusion matrix provides a detailed breakdown of the model's prediction accuracy for each class.
*   The **rows** represent the actual, true labels of the test samples.
*   The **columns** represent the labels predicted by the model.
*   The **diagonal cells** show the number of correct predictions. For example, all 20 `Normal` samples were correctly classified as `Normal`.
*   The **off-diagonal cells** show the number of incorrect predictions (misclassifications).

**Conclusion:** This is a perfect confusion matrix. All the values lie on the diagonal, and all off-diagonal cells are zero. This means the model achieved **100% accuracy** on the test set, making zero mistakes.

## 🚨 Detecting Power Line Cuts During Disasters

A critical application of this system is its ability to rapidly detect physical power line breaks, which are common during natural disasters like hurricanes, floods, or earthquakes.

**How it works:** A physical cut in a power line is known as an **open-conductor fault**. Our system simulates this exact scenario under the **`LT_Line_Cut`** fault class.

When a line is severed, the current flowing through that phase instantly drops to zero. Our model is trained to recognize the unique signature of this event through key features:
1.  **Massive Phase Imbalance:** The current in the healthy phases remains high while the cut phase drops to zero. This causes the `Phase_Imbalance` feature to skyrocket, which our model identifies as the single most important indicator of a fault.
2.  **Drastic RMS Drop:** The `RMS` value of the affected phase plummets, providing another clear signal.
3.  **Change in Zero Sequence Current:** The system becomes unbalanced, causing the `Zero_Sequence` current to deviate significantly from zero.

By deploying this model in a real-time monitoring system, utility companies can be alerted to an `LT_Line_Cut` fault the moment it happens. This allows them to pinpoint the location of a line break with much greater speed and accuracy, which is crucial for accelerating repair efforts, ensuring public safety, and restoring power during emergencies.

## 🚀 How to Run the Project

Follow these steps to set up and run the project on your local machine.

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and Activate a Virtual Environment** (Recommended)
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    Create a `requirements.txt` file with the following content:
    ```
    numpy
    pandas
    matplotlib
    scipy
    scikit-learn
    seaborn
    streamlit
    plotly
    joblib
    tqdm
    ```
    Then, run the installation command:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Generate the Dataset**
    Run the data generation script. This will create the `power_system_features.csv` file needed for training.
    ```bash
    python dataset_generator.py 
    ```
    *(Note: You will need to save the first code block from `Codes.md` as `dataset_generator.py`)*

5.  **Train the Machine Learning Model**
    Run the training script. This will train the models, save the best one (`power_system_fault_model.pkl` and related files), and generate the performance plots.
    ```bash
    python model_trainer.py
    ```
    *(Note: You will need to save the second code block from `Codes.md` as `model_trainer.py`)*

6.  **Launch the Monitoring Dashboard**
    Once the model is trained and saved, run the Streamlit application.
    ```bash
    streamlit run app.py
    ```
    *(Note: You will need to save the third code block from `Codes.md` as `app.py`)*

Your web browser should automatically open with the dashboard running. You can now interact with the real-time monitoring system