
import streamlit as st
import pandas as pd

from model_loader import load_approved_model

st.set_page_config(
    page_title="Visit with Us - Purchase Propensity",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Visit with Us")
st.subheader("Customer Purchase Propensity Prediction")

st.info(
    "Predicts purchase propensity for historical travel "
    "packages, not a specific new Wellness package."
)

# Token is configured through Streamlit app secrets.
token = st.secrets.get("MODEL_REGISTRY_PAT")

if not token:
    st.error("MODEL_REGISTRY_PAT is not configured.")
    st.stop()


@st.cache_resource
def get_model(access_token):
    # Download only the staging-approved model.
    return load_approved_model(access_token)


try:
    model, version = get_model(token)
except Exception as error:
    st.error(f"Unable to load approved model: {error}")
    st.stop()

st.caption(f"Approved staging model: {version}")

st.divider()

with st.form("customer_form"):

    st.subheader("Enter customer details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=35
        )

        income = st.number_input(
            "Monthly income", min_value=0, value=25000
        )

        trips = st.number_input(
            "Previous trips", min_value=0, value=3
        )

        persons = st.number_input(
            "Number of persons visiting",
            min_value=1, value=2
        )

        children = st.number_input(
            "Number of children visiting",
            min_value=0, value=0
        )

        city = st.selectbox(
            "City tier", [1, 2, 3]
        )

        occupation = st.selectbox(
            "Occupation",
            ["Salaried", "Small Business", "Large Business",
             "Freelancer"]
        )

    with col2:

        gender = st.selectbox(
            "Gender", ["Male", "Female"]
        )

        property_star = st.selectbox(
            "Preferred property star", [3, 4, 5]
        )

        marital_status = st.selectbox(
            "Marital status",
            ["Married", "Single", "Unmarried", "Divorced"]
        )

        passport = st.selectbox(
            "Has passport?", ["Yes", "No"]
        )

        own_car = st.selectbox(
            "Owns car?", ["Yes", "No"]
        )

        designation = st.selectbox(
            "Designation",
            ["Executive", "Manager", "Senior Manager",
             "AVP", "VP"]
        )

    submitted = st.form_submit_button(
        "Predict purchase propensity",
        type="primary"
    )

if submitted:

    if children > persons:
        st.error(
            "Children visiting cannot exceed total persons visiting."
        )
        st.stop()

    # Construct a one-row DataFrame using the trained schema.
    customer = pd.DataFrame([{
        "Age": age,
        "MonthlyIncome": income,
        "NumberOfTrips": trips,
        "NumberOfPersonVisiting": persons,
        "NumberOfChildrenVisiting": children,
        "CityTier": city,
        "Occupation": occupation,
        "Gender": gender,
        "PreferredPropertyStar": property_star,
        "MaritalStatus": marital_status,
        "Passport": 1 if passport == "Yes" else 0,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "Designation": designation
    }])

    # Use the same input feature ordering as training.
    if hasattr(model, "feature_names_in_"):
        customer = customer[
            list(model.feature_names_in_)
        ]

    positive_index = list(model.classes_).index(1)

    probability = model.predict_proba(customer)[
        0, positive_index
    ]

    st.divider()
    st.subheader("Prediction Result")

    st.metric(
        "Estimated purchase propensity",
        f"{probability:.1%}"
    )

    st.progress(float(probability))

    st.caption(
        "This is a model-estimated probability, "
        "not a guarantee of purchase."
    )
