# Med-Reserve: Synthetic Resource Planning Demo

A small Streamlit application for practicing a scikit-learn workflow, SQLite persistence, editable bed capacity, and a people list.

> **Demo only.** Training examples and labels are synthetic. This prototype is not clinically validated and must not be used to make patient triage, treatment, or real hospital bed-allocation decisions. Use fictional names and measurements only. Do not enter real personal or medical information.

## Run locally

Python 3.10 or newer is recommended.

1. Create and activate a virtual environment.
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Train and evaluate the demo models: `python train_model.py`
4. Start the app: `streamlit run app.py`

The training step creates `med_reserve_model.pkl` and `model_metrics.json`. The app creates `data/med_reserve.sqlite3` on first launch. Both generated artifacts are ignored by Git and can be regenerated locally.

Run the automated checks with `python -m unittest discover -s tests`.

## Using the app

- Enter a fictional name, age, SpO₂, heart rate, and comorbidity count, then choose **Add person**.
- The saved person appears in the list with a predicted risk category and admission outcome.
- Change total ICU slots and general beds in the capacity panel. Updates preserve occupied slots; the app rejects a reduction below current demo occupancy.
- The circular indicators show available versus total capacity.
- In Released history, select a previous person to prefill a fresh, editable assessment. This creates a separate record and leaves the old released record in history.
- You can permanently remove selected records from Released history. Active admissions cannot be removed this way.
- The Active admissions tab contains people with allocated beds. Discharge someone to move them into Released history and return the bed.
- The Released history tab keeps discharged records. The Not admitted tab shows saved assessments that could not receive a bed because capacity was full.

The database stores the entered name and values locally. Use fictional demo data only.

## What happens in the ML workflow

- A seeded generator creates 2,000 demo rows by default.
- A hand-written score based on age, SpO₂, and comorbidity count creates three target categories. Heart rate is included as an input but does not influence the generated target.
- Data is split into training and held-out test portions using stratification.
- Dummy Classifier, Logistic Regression, Decision Tree, and Random Forest are compared. The winner is selected using mean macro-F1 from five-fold cross-validation on the training portion.
- The selected model is evaluated on the held-out test portion. Training saves per-class precision/recall/F1, a confusion matrix, and model comparison metrics to `model_metrics.json`.

The score reflects how well models reproduce the synthetic rule. It is not evidence of clinical performance. Because the rule generates the labels, good scores are expected and do not show that the rule itself is valid.

The Logistic Regression pipeline scales its features. Tree models do not need scaling. Keeping preprocessing in a pipeline ensures inference uses the same learned preprocessing.

## Persistence and limitations

SQLite stores demo assessments and the current illustrative capacity of two resource groups. Assessment insertion and capacity decrement happen in one transaction. Releasing an allocation is also transactional and can only happen once. The initial snapshot preserves the original app's values: 12 of 15 ICU slots and 45 of 60 general/HDU slots are initially available. Capacity edits are persistent and retain the number of occupied slots.

The length-of-stay field from the original training data was removed from the app: the old version never trained a LOS model and instead used formulas or random values.

- Synthetic data has no clinical source, calibration, external validation, fairness assessment, or workflow approval.
- Resource labels are a software demonstration, not a recommendation for actual care.
- The app has no live hospital integration.

## Suggested interview explanation

“Med-Reserve is a Streamlit prototype using synthetic data to demonstrate a tabular classification and resource-tracking workflow. I compare a simple baseline and three classifiers with stratified cross-validation, select by macro-F1 on training folds, then report performance on a held-out synthetic test set. SQLite persists assessments and illustrative bed availability with transactions. Since labels come from a hand-written rule rather than clinical outcomes, these metrics only measure rule reproduction; the project is not clinically validated.”

## Interview questions to prepare

1. Why use a stratified train/test split, and what does stratification preserve?
2. Why use macro-F1 to compare classes? How would it differ from accuracy or weighted F1?
3. Why select the model using cross-validation on the training split and keep a separate test split?
4. What does the confusion matrix reveal that accuracy does not?
5. Why does Logistic Regression use scaling, while tree models generally do not?
6. Why is a Dummy Classifier useful as a baseline?
7. What does the synthetic label rule mean for the validity of the reported metrics?
8. Why is heart rate currently not informative to the model, and how should you address that in a real research design?
9. How does the SQLite transaction prevent concurrent assessments from overspending demo capacity?
10. What validation, privacy, fairness, calibration, clinical review, and governance would be necessary before any real healthcare use?

## Project layout

- `app.py`: Streamlit interface, capacity controls, circular availability indicators, and people list.
- `med_reserve/database.py`: SQLite schema, name and value validation, capacity updates, persistence, allocation, and release.
- `med_reserve/modeling.py`: synthetic data, model comparison, training, and metrics.
- `train_model.py`: repeatable model training entry point.
- `tests/`: standard-library unit tests.




