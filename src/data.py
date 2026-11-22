"""Data loading: CLINC150 (+ out-of-scope) and our hand-written home-appliance examples.

Every model in this project is trained on the same TRAIN split, tuned on the same
VAL split and evaluated on the same held-out TEST split built here.
"""
import csv
import json
import os
import random
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CLINC_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
CLINC_PATH = os.path.join(DATA_DIR, "data_full.json")
APPLIANCE_PATH = os.path.join(DATA_DIR, "appliance_examples.csv")

OOS = "oos"  # CLINC150's label for out-of-scope messages
APPLIANCE_OVERSAMPLE = 4  # we only have 12 train examples per appliance intent vs 100 for CLINC

# CLINC150 groups its 150 intents into 10 domains (15 intents each, see Larson et al., 2019).
DOMAINS = {
    "banking": "transfer transactions balance freeze_account pay_bill bill_balance bill_due interest_rate routing min_payment order_checks pin_change report_fraud account_blocked spending_history",
    "credit_cards": "credit_score report_lost_card credit_limit rewards_balance new_card application_status card_declined international_fees apr redeem_rewards credit_limit_change damaged_card replacement_card_duration improve_credit_score expiration_date",
    "kitchen_dining": "recipe restaurant_reviews calories nutrition_info restaurant_suggestion ingredients_list ingredient_substitution cook_time food_last meal_suggestion restaurant_reservation confirm_reservation how_busy cancel_reservation accept_reservations",
    "home": "shopping_list shopping_list_update next_song play_music update_playlist todo_list todo_list_update calendar calendar_update what_song order order_status reminder reminder_update smart_home",
    "auto_commute": "traffic directions gas gas_type distance current_location mpg oil_change_when oil_change_how jump_start uber schedule_maintenance last_maintenance tire_pressure tire_change",
    "travel": "book_flight book_hotel car_rental travel_suggestion travel_alert travel_notification carry_on timezone vaccines translate flight_status international_visa lost_luggage plug_type exchange_rate",
    "utility": "time alarm share_location find_phone weather text spelling make_call timer date calculator measurement_conversion flip_coin roll_dice definition",
    "work": "direct_deposit pto_request taxes payday w2 pto_balance pto_request_status next_holiday insurance insurance_change schedule_meeting pto_used meeting_schedule rollover_401k income",
    "small_talk": "greeting goodbye tell_joke where_are_you_from how_old_are_you what_is_your_name who_made_you thank_you what_can_i_ask_you what_are_your_hobbies do_you_have_pets are_you_a_bot meaning_of_life who_do_you_work_for fun_fact",
    "meta": "change_ai_name change_user_name cancel user_name reset_settings whisper_mode repeat no yes maybe change_language change_accent change_volume change_speed sync_device",
    # our own domain (hand-written examples in data/appliance_examples.csv)
    "appliance": "dishwasher_issue washer_issue fridge_issue warranty_repair",
}

# Which support team handles each domain.
TEAMS = {
    "banking": "Banking & Payments Team",
    "credit_cards": "Credit Card Team",
    "kitchen_dining": "Kitchen & Dining Team",
    "home": "Home & Orders Team",
    "auto_commute": "Auto & Commute Team",
    "travel": "Travel Team",
    "utility": "General Utility Team",
    "work": "HR & Payroll Team",
    "small_talk": "Chatbot (small talk)",
    "meta": "Account Settings Team",
    "appliance": "Appliance Technical Support",
}
HUMAN_AGENT = "Human agent"

INTENT_TO_DOMAIN = {i: d for d, s in DOMAINS.items() for i in s.split()}
# warranty / repair bookings go to their own team
SPECIAL_TEAMS = {"warranty_repair": "Warranty & Repair Team"}


def team_for(intent):
    if intent == OOS:
        return HUMAN_AGENT
    if intent in SPECIAL_TEAMS:
        return SPECIAL_TEAMS[intent]
    return TEAMS[INTENT_TO_DOMAIN[intent]]


def _download_clinc():
    if not os.path.exists(CLINC_PATH):
        os.makedirs(DATA_DIR, exist_ok=True)
        print("Downloading CLINC150 ...")
        urllib.request.urlretrieve(CLINC_URL, CLINC_PATH)


def _appliance_splits(seed=42):
    """12 train / 3 val / 5 test examples per appliance intent (fixed seed)."""
    by_intent = {}
    with open(APPLIANCE_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_intent.setdefault(row["intent"], []).append(row["text"])
    rng = random.Random(seed)
    train, val, test = [], [], []
    for intent, texts in sorted(by_intent.items()):
        texts = texts[:]
        rng.shuffle(texts)
        train += [(t, intent) for t in texts[:12]]
        val += [(t, intent) for t in texts[12:15]]
        test += [(t, intent) for t in texts[15:]]
    return train, val, test


def load_splits():
    """Returns dict with 'train', 'val', 'test' -> list of (text, intent)."""
    _download_clinc()
    with open(CLINC_PATH, encoding="utf-8") as f:
        d = json.load(f)
    a_train, a_val, a_test = _appliance_splits()
    splits = {
        "train": [tuple(x) for x in d["train"] + d["oos_train"]] + a_train * APPLIANCE_OVERSAMPLE,
        "val": [tuple(x) for x in d["val"] + d["oos_val"]] + a_val,
        "test": [tuple(x) for x in d["test"] + d["oos_test"]] + a_test,
    }
    return splits


def label_list():
    labels = sorted(INTENT_TO_DOMAIN) + [OOS]
    assert len(labels) == 155, len(labels)  # 150 CLINC + 4 appliance + oos
    return labels


if __name__ == "__main__":
    s = load_splits()
    for k, v in s.items():
        n_oos = sum(1 for _, y in v if y == OOS)
        print(f"{k:5s}: {len(v):6d} examples ({n_oos} out-of-scope)")
    assert all(y in INTENT_TO_DOMAIN or y == OOS for v in s.values() for _, y in v)
    print("labels:", len(label_list()))
