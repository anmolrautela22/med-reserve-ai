# Med-Reserve

Med-Reserve is a Python and Streamlit project for recording patient measurements and tracking ICU and general bed availability. It uses SQLite to save assessments and scikit-learn to assign each record a risk category.

## Live app

Open Med-Reserve: https://medres.streamlit.app/

It uses synthetic training data. Its predictions have not been clinically validated. Records are stored in a local SQLite database.

## What it does

- Add a person with their name, age, SpO₂, heart rate, and comorbidity count.
- See current ICU and general bed availability.
- Change the total capacity for either type of bed.
- View active admissions, released records, and assessments that could not be allocated a bed.
- Discharge an active admission, create a new record from a released one, or remove a released record from the saved list.

SQLite keeps the assessment history and bed counts when the app restarts. Capacity changes keep the existing number of occupied beds. The database stores the information entered in the form, so use fictional information rather than real patient details.

## Model

The training script creates 2,000 synthetic records by default. A simple formula using age, SpO₂, and comorbidity count assigns the category labels. Heart rate is part of the model input, but it does not affect how those labels are generated.

The script compares a majority-class baseline, Logistic Regression, Decision Tree, and Random Forest. It selects a model using five-fold cross-validation on the training data, then reports results on a separate test set. The output includes per-category scores, a confusion matrix, and model comparisons.

These scores show how closely the models reproduce the formula used to generate the labels. They do not show how well the system would work with real clinical outcomes. The project has not been validated for clinical use, and its predictions should not guide care.

The project does not predict length of stay or connect to a real hospital system.

## Files

- `app.py` contains the Streamlit interface.
- `med_reserve/database.py` manages SQLite records, bed capacity, admissions, and releases.
- `med_reserve/modeling.py` generates data, trains models, and produces evaluation results.
- `train_model.py` starts model training.
- `tests/` contains the automated checks.

