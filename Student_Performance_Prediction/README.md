# Student Performance Prediction Using Multiple Linear Regression

## Team Members

| Name | Roll Number |
|------|-------------|
| V. Laxmi Ganga Bhavani | 25RBAID0026 |
| R. Santhosh | 25RBAIM0076 |
| T. Manideep | 25RBAIM0069 |

**Guide:** Dr. M. Praveen Sir  
**Department:** CSE (AI & ML), MNR University, School of Engineering & Technology and AI  
**Academic Year:** 2026-2027

---

## Project Description

Students often receive feedback only after final exams, making it too late for improvement. This project builds a **Multiple Linear Regression** model that predicts a student's final exam marks based on 5 input factors — study hours, attendance, previous marks, assignment scores, and sleep hours. This allows faculty to identify at-risk students early and provide timely academic support.

---

## Dataset

- **Name:** Student Performance Dataset
- **Records:** 500 student records, 6 columns
- **Location:** Included in the `dataset/` folder — [student_performance_dataset.csv](dataset/student_performance_dataset.csv)

| Feature | Description | Range |
|---------|-------------|-------|
| Study_Hours | Daily study hours | 1.0 - 9.4 |
| Attendance | Class attendance percentage | 60.2% - 99.0% |
| Previous_Marks | Prior exam scores | 40.1 - 94.9 |
| Assignment_Score | Assignment grades | 50.1 - 97.9 |
| Sleep_Hours | Daily sleep hours | 4.5 - 9.0 |
| Final_Marks (Target) | Final exam marks | 58.0 - 100.0 |

---

## Algorithm Used

**Multiple Linear Regression (Ordinary Least Squares)**

The trained model equation:

```
Final_Marks = 5.4883 + 2.0181(Study_Hours) + 0.2754(Attendance) + 0.3742(Previous_Marks) + 0.1841(Assignment_Score) + 0.7488(Sleep_Hours)
```

---

## How to Install and Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Student-Performance-Prediction.git
cd Student-Performance-Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Python script
```bash
python Student_Performance_Code/student_performance_prediction.py
```

### 4. Or open the Jupyter Notebook
```bash
jupyter notebook Student_Performance_Code/student_performance_prediction.ipynb
```

---

## Results

### Model Performance on Test Data (100 unseen records)

| Metric | Value |
|--------|-------|
| R-squared (R2) | 0.9444 (94.44%) |
| Mean Absolute Error (MAE) | 1.5975 marks |
| Root Mean Squared Error (RMSE) | 2.0606 marks |
| Mean Squared Error (MSE) | 4.2462 |

### Training vs Testing Comparison

| Set | R2 Score | MAE |
|-----|----------|-----|
| Training (400 records) | 0.9515 (95.15%) | 1.6920 |
| Testing (100 records) | 0.9444 (94.44%) | 1.5975 |

The small gap (0.71%) confirms **no overfitting**.

### Feature Importance (Learned Weights)

| Feature | Weight | Meaning |
|---------|--------|---------|
| Study_Hours | +2.0181 | Each extra hour of study adds ~2 marks |
| Sleep_Hours | +0.7488 | Each extra hour of sleep adds ~0.75 marks |
| Previous_Marks | +0.3742 | Higher past scores predict higher final marks |
| Attendance | +0.2754 | Better attendance leads to better scores |
| Assignment_Score | +0.1841 | Good assignment performance helps final marks |

### Graphs

All plots are in the `Student_Performance_Figures/` folder:

- Correlation Heatmap
- Actual vs Predicted Scatter Plot
- Residual Error Plot
- Feature Coefficients Bar Chart
- Dataset Feature Distributions
- System Architecture Diagram
- ML Life Cycle Diagram

### Sample Predictions

| Student Type | Predicted Marks | Classification |
|-------------|----------------|----------------|
| High Performer | 100.00 | Outstanding (A+) |
| Average Student | 81.37 | Good (A/B) |
| At-Risk Student | 57.30 | Average / Pass (C) |

---

## Demo

To run the demo locally:

```bash
python Student_Performance_Code/student_performance_prediction.py
```

This runs the complete ML pipeline: data loading, feature analysis, model training, evaluation metrics, and 5 test case predictions.

---

## Project Structure

```
Student_Performance_Prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── Student_Performance_Prediction_Project_Report.docx
├── Student_Performance_Prediction_Project_Report.pdf
├── Student_Performance_Prediction_Mini_Project.pptx
├── dataset/
│   └── student_performance_dataset.csv
├── Student_Performance_Code/
│   ├── student_performance_prediction.py
│   └── student_performance_prediction.ipynb
└── Student_Performance_Figures/
    ├── actual_vs_predicted.png
    ├── correlation_heatmap.png
    ├── dataset_feature_representation.png
    ├── feature_coefficients.png
    ├── linear_regression_conceptual.png
    ├── ml_life_cycle.png
    ├── project_workflow.png
    ├── residual_plot.png
    └── system_architecture.png
```
