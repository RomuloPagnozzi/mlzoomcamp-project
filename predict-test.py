import requests

url = 'http://localhost:9696/predict'

customer = {
    "income": 0.8,
    "name_email_similarity": 0.1117106829885132,
    "prev_address_months_count": -1,
    "current_address_months_count": 350,
    "customer_age": 30,
    "days_since_request": 0.0066701837308879,
    "intended_balcon_amount": -0.6745417333311727,
    "payment_type": "AD",
    "zip_count_4w": 1280,
    "velocity_6h": 8156.887604248211,
    "velocity_24h": 5418.824044563015,
    "velocity_4w": 5077.1695244916245,
    "bank_branch_count_8w": 12,
    "date_of_birth_distinct_emails_4w": 9,
    "employment_status": "CA",
    "credit_risk_score": 157,
    "email_is_free": 0,
    "housing_status": "BA",
    "phone_home_valid": 0,
    "phone_mobile_valid": 1,
    "bank_months_count": 1,
    "has_other_cards": 0,
    "proposed_credit_limit": 200.0,
    "foreign_request": 0,
    "source": "INTERNET",
    "session_length_in_minutes": 1.044920255755289,
    "device_os": "linux",
    "keep_alive_session": 1,
    "device_distinct_emails_8w": 1,
    "month": 4
}

response = requests.post(url, json=customer).json()
print(response)