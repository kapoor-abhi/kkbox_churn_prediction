import json
import random
from datetime import datetime, timedelta

# File path
file_path = "data/live_traffic.jsonl"

# Base structure (from your provided example)
def generate_record(i):
    # Simulate some drift/randomness
    is_drifted = i > 40  # Make the last 10 records significantly different
    
    return {
        "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
        "input": {
            "msno": f"test_user_{100 + i}",
            "city": random.choice([1, 4, 5, 13, 22]) if not is_drifted else 1,
            "bd": random.randint(18, 45) if not is_drifted else random.randint(80, 100), # Drift: older ages
            "gender": random.choice(["male", "female"]),
            "registered_via": random.choice([3, 4, 7, 9]),
            "total_transactions": random.randint(1, 20),
            "total_payment": random.uniform(99.0, 1200.0) if not is_drifted else random.uniform(0.0, 50.0), # Drift: lower payments
            "total_cancel_count": random.choice([0, 0, 1]),
            "promo_transaction_count": random.choice([0, 1]),
            "avg_plan_days": 30.0,
            "days_since_last_transaction": random.randint(0, 60),
            "total_secs_played": random.uniform(500.0, 50000.0),
            "total_unique_songs": random.randint(10, 500),
            "total_songs_played": random.randint(20, 1000),
            "total_songs_100_percent": random.randint(5, 500),
            "active_days": random.randint(1, 30),
            "active_days_first_half": random.randint(1, 15),
            "active_days_second_half": random.randint(0, 15),
            "total_secs_first_half": random.uniform(200.0, 25000.0),
            "total_secs_second_half": random.uniform(0.0, 25000.0)
        },
        "prediction_prob": random.uniform(0.1, 0.99),
        "prediction_class": random.choice([0, 1])
    }

# Append 50 records
with open(file_path, "a") as f:
    for i in range(50):
        record = generate_record(i)
        f.write(json.dumps(record) + "\n")

print(f"✅ Successfully appended 50 records to {file_path}")