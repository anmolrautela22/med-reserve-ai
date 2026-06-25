# 🏥 Med-Reserve: AI-Driven Patient Triage & Bed Resource Planner

Med-Reserve is an end-to-end clinical operations platform engineered to optimize critical care resource allocation and automate emergency room patient triage. By leveraging an embedded machine learning inference engine, the application tracks real-time patient physiological vitals to calculate risk stratifications and dynamically update live hospital bed inventory.

## 📊 System Architecture & Workflow
The application is structured into a modular, decoupled architecture to replicate production environments:

1. **Clinical Triage Pipeline:** Captures real-time patient biometric inputs (Age, Heart Rate, Oxygen Saturation, Comorbidities).
2. **Machine Learning Core:** Forwards inputs through a Scikit-learn preprocessing standardizer and executes a multi-class predictive classification tree.
3. **Stateful Resource Monitor:** Processes inference outputs to calculate Estimated Length of Stay (LOS), dynamically reserves local bed assets, and triggers immediate care routing workflows.

## 🛠️ Technology Stack
- **Backend & Data Science:** Python, Scikit-learn, Pandas, NumPy
- **Frontend & State Management:** Streamlit
- **Model Serialization:** Pickle (Binary Artifacts)

## 📈 Engineering Design Decisions
- **Pipeline Pipeline Modularity:** Separated data engineering and model selection training (`train_model.py`) from live inference routines (`app.py`) via pipeline compilation to prevent telemetry data leakage.
- **Robust Feature Scaling:** Implemented a `StandardScaler` to uniformize varying metrics across physiological ranges, ensuring stable gradient bounds for the underlying statistical models.
- **State-Preserved Resource Tracking:** Utilized transactional session state caches to prevent race conditions during mock database metric modifications during live input cycles.