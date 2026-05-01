# Stress Reduction Through Mindfulness Interventions for Tech Industry Professionals
### UC Berkeley ML/AI Professional Certificate — Capstone Project

**Author:** Rupleena Chhabra
**LinkedIn:** [linkedin.com/in/rupleena](https://www.linkedin.com/in/rupleena/)
**GitHub:** [github.com/rupleena/ml-ai-intervention-outcome-study](https://github.com/rupleena/ml-ai-intervention-outcome-study)

---

## Executive Summary

Tech industry professionals face disproportionately high rates of chronic stress and burnout, driven by always-on work culture, high cognitive demands, and sustained performance pressure. While mindfulness-based interventions are increasingly embedded in corporate wellness programs, the evidence base for which specific practices reduce stress — and for which individuals — remains limited.

This project applies machine learning classification, clustering, and feature analysis to explore whether a tech worker's role, work conditions, and engagement patterns can indicate which mindfulness interventions may be worth trying for someone with their profile.

The goal is practical: provide an evidence-informed baseline for recommending intervention approaches based on observable profile and engagement signals, while being transparent about current predictive limits.

---

## Research Question

> Can we predict whether a mindfulness intervention will meaningfully reduce stress for a tech industry professional, based on their role, work conditions, and engagement patterns?

---

## Repository Structure

```
ml-ai-intervention-outcome-study/
├── README.md                          ← This file
├── data/
│   └── intervention_response_tech.csv ← Dataset (1,200 rows, 19 features)
├── intervention_outcome_study.ipynb   ← Single final notebook (EDA + Modelling + Results)
└── report/
    └── intervention_report.pdf        ← Full capstone report
```

---

## Notebook Overview

📓 [`intervention_outcome_study.ipynb`](./intervention_outcome_study.ipynb)
📄 [`report/intervention_report.pdf`](./report/intervention_report.pdf)

The notebook is structured as a single end-to-end analysis across 12 sections:

| Section | Description |
|---|---|
| 1. Imports & Setup | Libraries and configuration |
| 2. Data Loading & Inspection | Shape, dtypes, missing values, class distribution |
| 3. Exploratory Data Analysis | Intervention rates, role/seniority, engagement patterns, correlations |
| 4. Data Cleaning & Feature Engineering | Consistency score, work pressure index, experience level, encoding |
| 5. Train / Test Split & Scaling | Stratified 80/20 split, StandardScaler |
| 6. Clustering — Stress-Risk Profiles | K-Means k=3, elbow method, PCA visualisation, cluster profiling |
| 7. Baseline Model | Majority-class dummy classifier |
| 8. Hyperparameter Tuning | GridSearchCV for LR, KNN, Decision Tree, SVM |
| 9. Final Model Comparison | Results table, bar charts, ROC curves, confusion matrices |
| 10. Feature Importance | LR coefficients, DT importances, tree structure |
| 11. Cross-Validation Summary | 5-fold CV comparison across all models |
| 12. Key Findings & Recommendations | Summary table, written findings, individual guidance |

---

## Dataset

**File:** `data/intervention_response_tech.csv`
**Rows:** 1,200 | **Features:** 19 | **Target:** `stress_reduced` (binary: 1 = meaningful stress reduction, 0 = no meaningful reduction)

**Target definition:** A reduction of ≥3 points on the DASS stress scale (pre vs post intervention) is treated as a meaningful clinical improvement. Positive class rate: ~38%.

| Feature Group | Features |
|---|---|
| Worker profile | `role`, `seniority`, `company_size`, `work_type` |
| Work stressors | `weekly_hours`, `oncall_freq`, `meetings_per_day` |
| Intervention | `intervention`, `sessions_per_week`, `avg_session_min`, `completion_rate`, `practice_time`, `weeks_of_practice` |
| Psychometric | `mindfulness_baseline`, `stress_pre` |
| Engineered | `consistency_score`, `work_pressure_idx`, `experience_level` |
| Target | `stress_reduced` |

**Data sources this dataset draws from:**
- [DASS Responses — Kaggle](https://www.kaggle.com/datasets/lucasgreenwell/depression-anxiety-stress-scales-responses)
- [Mental Health in Tech Survey — Kaggle / OSMI](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
- [RU Mindful Dataset — Rowan University](https://rdw.rowan.edu/datasets/2/)
- [Kentucky Inventory of Mindfulness Skills — Kaggle](https://www.kaggle.com/datasets/lucasgreenwell/kentucky-inventory-of-mindfulness-skills-responses)

> Note: This project dataset is a curated/engineered research dataset assembled from these public sources and transformed for the capstone analysis workflow.

---

## Model Results

| Model | Test Accuracy | ROC-AUC | F1-Score | CV AUC |
|---|---|---|---|---|
| Majority-Class Baseline | 0.6208 | 0.5000 | 0.0000 | — |
| Logistic Regression (tuned) | 0.5208 | 0.5193 | **0.4335** | 0.5886 ± 0.0261 |
| KNN (tuned) | 0.5750 | 0.4688 | 0.1774 | 0.5830 ± 0.0478 |
| Decision Tree (tuned) | 0.6083 | 0.5035 | 0.0600 | 0.5263 ± 0.0330 |
| **SVM (tuned)** | 0.5250 | 0.5047 | 0.4000 | **0.6007 ± 0.0269** |

**Primary metric: ROC-AUC** — chosen over accuracy due to class imbalance (~38% positive class). A model predicting the majority class every time scores 62% accuracy but an F1 of zero.

**Best overall:** SVM achieves the highest cross-validation AUC (0.6007). Logistic Regression achieves the best test F1-Score (0.4335) and remains the most interpretable.

---

## Key Findings

**1. Consistency is the strongest predictor**
The engineered `consistency_score` — combining sessions per week, completion rate, and weeks of practice — is the top positive predictor. Regular, completed practice sustained over time drives outcomes more than session length or intervention type alone.

**2. Intervention type matters**
Breathing exercises and body scans show the highest stress reduction rates. Journaling and sleep practices show the lowest. Guided meditation falls in the middle — notable given it is the most commonly recommended starting point.

**3. Work environment limits effectiveness**
`oncall_freq` and `work_pressure_idx` are the strongest negative predictors. High structural stress actively limits the effectiveness of self-administered interventions.

**4. Three distinct workforce segments**
K-Means clustering (k=3) identified three stress-risk profiles with meaningfully different reduction rates:
- **Consistent Practitioners** — ~48% reduction rate, most responsive
- **Moderate Engagers** — ~39% reduction rate, strongest opportunity for improvement
- **High-Burnout / Low-Response** — ~32% reduction rate, may need structural or clinical-level support beyond self-administered practices

**5. Honest model performance**
All models show modest but consistent improvement above the random baseline on ROC-AUC. This reflects the genuine difficulty of predicting individual stress outcomes from survey data and establishes a transparent baseline for future work with richer longitudinal and physiological data.

---

## Next Steps

- Apply SMOTE or class-weight balancing to further improve recall on the positive class
- Train segment-specific models per cluster and compare against the global model
- Incorporate longitudinal engagement data (daily tracking over weeks) for richer features
- Add physiological signals (HRV, sleep quality) as objective stress proxies
- Explore ensemble methods (Random Forest, XGBoost) as natural extensions

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/rupleena/ml-ai-intervention-outcome-study.git
cd ml-ai-intervention-outcome-study

# (Optional but recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# Launch notebook
jupyter notebook intervention_outcome_study.ipynb
```

If you already use Anaconda/Miniconda:

```bash
conda create -n mindfulness-capstone python=3.11 -y
conda activate mindfulness-capstone
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook intervention_outcome_study.ipynb
```

---

## Reproducibility Notes

- Set random seeds in the notebook for train/test split, K-Means, and model training where supported.
- Keep package versions stable for direct metric comparison across runs.
- Use stratified splitting for classification tasks to preserve class balance.

---

## Citation & Rights

© 2026 Rupleena Chhabra. All rights reserved.
This work is the original work of Rupleena Chhabra. Unauthorized reproduction, distribution, or use of this material without explicit written permission from the author is prohibited.
