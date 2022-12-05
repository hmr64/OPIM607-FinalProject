import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import streamlit as st
# streamlit run final.py

s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    return np.where(x == 1, 1, 0)

ss = pd.DataFrame({
    "sm_li":clean_sm(s.web1h),
    "income":np.where(s.income < 10, s.income, np.nan),
    "education":np.where(s.educ2 < 9, s.educ2, np.nan),
    "parent":np.where(s.par == 1, True, np.where(s.par == 2, False, np.nan)),
    "married":np.where(s.marital == 1, True, np.where(s.marital < 8, False, np.nan)),
    "female":np.where(s.gender == 2, True, np.where(s.gender < 4, False, np.nan)),
    "age":np.where(s.age < 98, s.age, np.nan)
})

# Clean data
ss = ss.dropna()

# Target (y) and feature(s) selection (X)
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987) # set for reproducibility


# Fit training data
lr = LogisticRegression(class_weight="balanced")
lr.fit(X_train, y_train)


###### Streamlit code

st.markdown("# LinkedIn User Prediction")

st.markdown("### Fill out the following information about yourself, \
	and our algorithm will predict if you are a LinkedIn user")

# income

income_dict = {
	"< 10k":1,
	"10k - 20k":2,
	"20k - 30k":3,
	"30k - 40k":4,
	"40k - 50k":5,
	"50k - 75k":6,
	"75k - 100k":7,
	"100k - 150k":8,
	" > 150k":9
}

income = st.selectbox(label="Income", options=(list(income_dict.keys())))

# education

education_dict = {
	"Less than high school":1,
	"High school incomplete":2,
	"High school graduate":3,
	"Some college, no degree":4,
	"Two-year associate degree":5,
	"Four-year college or university/Bachelor's degree":6,
	"Some postgraduate or professional schooling, not postgrad degree":7,
	"Postgraduate or professional degree, including MA, MS, PhD, MD, JD":8
}

education = st.selectbox(label="Education", options=(list(education_dict.keys())))

# parent

parent = False
if st.checkbox("Are you a parent?"):
    parent = True
else:
	parent = False

# married

married = False
if st.checkbox("Are you married?"):
    married = True
else:
	married = False

# female

female = False
if st.checkbox("Are you female?"):
    female = True
else:
	female = False

# age

age = st.slider("Age", min_value = 1, max_value = 97)


# Predict
if st.button("Predict!"):
	# New data for features
	person = [income_dict[income], education_dict[education], parent, married, female, age]

	# Predict class, given input features
	predicted_class = lr.predict([person])
	st.write("We predict that you are", "" if predicted_class == 1 else "not", "a LinkedIn user")

	# Generate probability of positive class (=1)
	probs = lr.predict_proba([person])
	st.write(f"Probability that you are a LinkedIn user: {round(probs[0][1] * 100, 2)}%")



