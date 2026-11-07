
import streamlit as st
import pandas as pd
import joblib

from datetime import date

st.set_page_config(
    page_title="Hotel Revenue Protection",
    page_icon="🏨",
    layout="wide"
)

# 2. Load saved model and thresholds
@st.cache_resource
def load_model():
    return joblib.load("hotel_cancellation_model.pkl")

model = load_model()

thresholds = joblib.load("deposit_thresholds.pkl")

st.title("🏨 Hotel Revenue Protection")
st.subheader("AI-Powered Booking Cancellation Predictor")

st.write(
    "Predict booking cancellation risk and receive "
    "an intelligent deposit recommendation."
)

st.divider()

# 4. Booking form
st.header("Enter Booking Information")

with st.form("booking_form"):

    st.subheader("Booking Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        hotel = st.selectbox(
            "Hotel Type",
            ["City Hotel", "Resort Hotel"]
        )

        arrival_date = st.date_input(
            "Arrival Date",
            value=date(2026, 12, 1)
        )

        lead_time = st.number_input(
            "Lead Time (Days)",
            min_value=0,
            value=120
        )

    with col2:
        adults = st.number_input(
            "Adults",
            min_value=1,
            value=2
        )

        children = st.number_input(
            "Children",
            min_value=0,
            value=0
        )

        babies = st.number_input(
            "Babies",
            min_value=0,
            value=0
        )

    with col3:
        weekend_nights = st.number_input(
            "Weekend Nights",
            min_value=0,
            value=2
        )

        week_nights = st.number_input(
            "Week Nights",
            min_value=0,
            value=3
        )

        adr = st.number_input(
            "Average Daily Rate (ADR)",
            min_value=0.0,
            value=120.0
        )

    st.divider()
    st.subheader("Customer and Booking Information")

    col4, col5, col6 = st.columns(3)

    with col4:
        meal = st.selectbox(
            "Meal Type",
            ["BB", "FB", "HB", "SC", "Undefined"]
        )

        country = st.text_input(
            "Country Code",
            value="PRT",
            help="Enter a country code, for example PRT or GBR."
        )

        market_segment = st.selectbox(
            "Market Segment",
            [
                "Online TA",
                "Offline TA/TO",
                "Direct",
                "Groups",
                "Corporate",
                "Complementary",
                "Aviation",
                "Undefined"
            ]
        )

    with col5:
        distribution_channel = st.selectbox(
            "Distribution Channel",
            ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
        )

        reserved_room_type = st.selectbox(
            "Reserved Room Type",
            ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
        )

        customer_type = st.selectbox(
            "Customer Type",
            ["Transient", "Transient-Party", "Contract", "Group"]
        )

    with col6:
        is_repeated_guest = st.selectbox(
            "Repeated Guest?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        previous_cancellations = st.number_input(
            "Previous Cancellations",
            min_value=0,
            value=0
        )

        previous_bookings_not_canceled = st.number_input(
            "Previous Successful Bookings",
            min_value=0,
            value=0
        )

    st.divider()
    st.subheader("Additional Requests")

    col7, col8 = st.columns(2)

    with col7:
        parking = st.number_input(
            "Required Parking Spaces",
            min_value=0,
            value=0
        )

    with col8:
        special_requests = st.number_input(
            "Total Special Requests",
            min_value=0,
            value=1
        )

    submitted = st.form_submit_button(
        "Predict Cancellation Risk",
        use_container_width=True
    )

# 5. Make prediction
if submitted:

    if not country.strip():
        st.error("Please enter a country code.")
        st.stop()

    total_nights = weekend_nights + week_nights
    total_guests = adults + children + babies

    booking_data = {
        "hotel": hotel,
        "lead_time": lead_time,
        "stays_in_weekend_nights": weekend_nights,
        "stays_in_week_nights": week_nights,
        "adults": adults,
        "children": children,
        "babies": babies,
        "meal": meal,
        "country": country.strip().upper(),
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": is_repeated_guest,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "reserved_room_type": reserved_room_type,
        "customer_type": customer_type,
        "adr": adr,
        "required_car_parking_spaces": parking,
        "total_of_special_requests": special_requests,
        "arrival_day_of_week": arrival_date.weekday(),
        "arrival_month_num": arrival_date.month,
        "total_nights": total_nights,
        "total_guests": total_guests
    }

    # Keep the exact training feature order
    input_df = pd.DataFrame(
        [booking_data],
        columns=model.feature_names_in_
    )

    # Predict cancellation probability
    probability = model.predict_proba(input_df)[0, 1]

    # Classification threshold
    predicted_class = int(probability >= 0.50)

    # Deposit recommendation
    if probability < thresholds["no_deposit"]:
        recommendation = "No Deposit"
        risk_level = "Low Risk"

    elif probability < thresholds["partial_deposit"]:
        recommendation = "Partial Deposit"
        risk_level = "Medium Risk"

    else:
        recommendation = "Full Deposit"
        risk_level = "High Risk"

    # 6. Display results
    st.divider()
    st.header("Prediction Results")

    result1, result2, result3 = st.columns(3)

    with result1:
        st.metric(
            "Cancellation Probability",
            f"{probability * 100:.2f}%"
        )

    with result2:
        st.metric(
            "Predicted Status",
            "Canceled" if predicted_class == 1 else "Not Canceled"
        )

    with result3:
        st.metric(
            "Risk Level",
            risk_level
        )

    st.subheader("Recommended Deposit Policy")

    if recommendation == "No Deposit":
        st.success("Recommended Policy: No Deposit")

    elif recommendation == "Partial Deposit":
        st.warning("Recommended Policy: Partial Deposit")

    else:
        st.error("Recommended Policy: Full Deposit")

    st.progress(float(probability))

    st.write(
        f"Estimated cancellation risk: {probability * 100:.2f}%"
    )

    st.info(
        "This recommendation is based on historical booking patterns. It is a decision-support tool, not a guarantee that a deposit "
        "will prevent cancellation."
    )
