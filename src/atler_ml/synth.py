"""Synthetic Indian bank statements with ground truth (v2: per-merchant spread).

ATLER keeps money on the phone and real statements are private, so there is
no labelled dataset to learn from. This generator writes statements that look
like the ones ATLER imports (UPI / POS / NACH narrations, legal names that
differ from brand names, posting delays, price rises, cancelled plans) and
records the truth behind every row: its category, which subscription it
belongs to, and whether it was planted as an unusual spend.

Everything is driven by one seed, so a benchmark run is reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

CATEGORIES = [
    "Food", "Groceries", "Transport", "Entertainment", "Utilities",
    "Shopping", "Health", "Education", "Productivity", "Transfers",
]

# (spellings seen in narrations, category, typical amount in rupees).
# Several spellings are legal names, not brands (Bundl = Swiggy, Eternal =
# Zomato): keyword rules miss those, a model that has seen your filing doesn't.
EVERYDAY: list[tuple[list[str], str, float]] = [
    (["SWIGGY", "Swiggy Limited", "BUNDL TECHNOLOGIES"], "Food", 320),
    (["ZOMATO", "ZOMATO LTD", "ETERNAL LIMITED"], "Food", 350),
    (["SRI SARAVANA BHAVAN", "SARAVANA BHAVAN"], "Food", 180),
    (["A2B ADYAR ANANDA BHAVAN", "ADYAR ANANDA"], "Food", 210),
    (["JAVA GREEN CANTEEN", "JAVA CANTEEN SRM"], "Food", 90),
    (["STARBUCKS", "TATA STARBUCKS"], "Food", 420),
    (["DOMINOS PIZZA", "JUBILANT FOODWORKS"], "Food", 450),
    (["BLINKIT", "GROFERS INDIA"], "Groceries", 380),
    (["ZEPTO", "KIRANAKART TECHNOLOGIES"], "Groceries", 340),
    (["BIGBASKET", "SUPERMARKET GROCERY SUPPLIES"], "Groceries", 900),
    (["DMART", "AVENUE SUPERMARTS"], "Groceries", 1400),
    (["ANNA STORES", "SRI MURUGAN STORES"], "Groceries", 160),
    (["UBER", "UBER INDIA SYSTEMS"], "Transport", 260),
    (["RAPIDO", "ROPPEN TRANSPORTATION"], "Transport", 90),
    (["OLA", "ANI TECHNOLOGIES"], "Transport", 240),
    (["CHENNAI METRO RAIL", "CMRL"], "Transport", 50),
    (["IRCTC", "INDIAN RAILWAY CATERING"], "Transport", 780),
    (["INDIAN OIL", "IOCL FUEL STATION"], "Transport", 500),
    (["BOOKMYSHOW", "BIGTREE ENTERTAINMENT"], "Entertainment", 450),
    (["PVR CINEMAS", "PVR INOX"], "Entertainment", 380),
    (["STEAM", "VALVE STEAM"], "Entertainment", 600),
    (["TNEB", "TANGEDCO"], "Utilities", 1100),
    (["AMAZON", "AMAZON SELLER SERVICES", "AMZN MKTP"], "Shopping", 900),
    (["FLIPKART", "FLIPKART INTERNET"], "Shopping", 1100),
    (["MYNTRA", "MYNTRA DESIGNS"], "Shopping", 1300),
    (["DECATHLON", "DECATHLON SPORTS"], "Shopping", 1500),
    (["APOLLO PHARMACY", "APOLLO PHARMACIES"], "Health", 340),
    (["TATA 1MG", "1MG TECHNOLOGIES"], "Health", 420),
    (["MEDPLUS", "OPTIVAL HEALTH"], "Health", 260),
    (["HIGGINBOTHAMS", "HIGGINBOTHAMS BOOKS"], "Education", 450),
    (["SRM IST FEES", "SRM INSTITUTE"], "Education", 2500),
    (["XEROX POINT", "SRI LAKSHMI XEROX"], "Education", 40),
]

# How much a merchant's amounts vary (sigma of log amount). Fixed fares and
# fees barely move; marketplaces and train tickets swing a lot. Everything
# else, 0.45. (Before v2 every merchant used 0.45, which hid the difference
# between "6x at the metro" and "6x at Amazon".)
SPREAD = {
    "CHENNAI METRO RAIL": 0.12, "SRM IST FEES": 0.08, "XEROX POINT": 0.2, "JAVA GREEN CANTEEN": 0.2,
    "RAPIDO": 0.3, "TNEB": 0.3, "STARBUCKS": 0.25,
    "AMAZON": 0.8, "FLIPKART": 0.8, "MYNTRA": 0.7, "DECATHLON": 0.7, "IRCTC": 0.75, "DMART": 0.6,
    "BIGBASKET": 0.6, "INDIAN OIL": 0.5, "UBER": 0.5,
}
DEFAULT_SPREAD = 0.45

# Subscriptions: (spellings, category, price in rupees, cycle in days or "month"/"year").
SUBSCRIPTIONS: list[tuple[list[str], str, float, int | str]] = [
    (["NETFLIX", "NETFLIX.COM"], "Entertainment", 199, "month"),
    (["SPOTIFY", "SPOTIFY INDIA"], "Entertainment", 119, "month"),
    (["YOUTUBE PREMIUM", "GOOGLE YOUTUBE"], "Entertainment", 149, "month"),
    (["JIOHOTSTAR", "STAR INDIA"], "Entertainment", 299, 91),
    (["AMAZON PRIME", "PRIME VIDEO"], "Entertainment", 1499, "year"),
    (["JIO PREPAID", "RELIANCE JIO"], "Utilities", 349, 28),
    (["AIRTEL PREPAID", "BHARTI AIRTEL"], "Utilities", 379, 28),
    (["ACT FIBERNET", "ATRIA CONVERGENCE"], "Utilities", 799, "month"),
    (["GOOGLE ONE", "GOOGLE STORAGE"], "Productivity", 130, "month"),
    (["CHATGPT", "OPENAI"], "Productivity", 1999, "month"),
    (["NOTION LABS", "NOTION"], "Productivity", 830, "month"),
    (["GITHUB", "GITHUB INC"], "Productivity", 340, "month"),
    (["CULTFIT", "CUREFIT HEALTHCARE"], "Health", 1299, "month"),
    (["DUOLINGO", "DUOLINGO INC"], "Education", 3499, "year"),
    (["COURSERA", "COURSERA INC"], "Education", 4100, "month"),
    (["SWIGGY ONE", "BUNDL ONE MEMBERSHIP"], "Food", 99, 91),
]

FIRST = ["RAMESH", "PRIYA", "ARJUN", "DIVYA", "KARTHIK", "SNEHA", "VIGNESH", "ANANYA", "RAHUL", "MEERA",
         "SURESH", "LAKSHMI", "ADITYA", "KAVYA", "HARISH", "NIVEDHA"]
LAST = ["KUMAR", "SHARMA", "IYER", "REDDY", "NAIR", "RAJAN", "SINGH", "PILLAI", "MENON", "GUPTA"]
BANKS = ["HDFC", "ICIC", "SBIN", "UTIB", "KKBK", "YESB"]
HANDLES = ["ybl", "okaxis", "okhdfcbank", "paytm", "oksbi", "ibl"]
CITIES = ["CHENNAI", "KATTANKULATH", "BANGALORE", "MUMBAI"]

# Shops near you that no one else's data has seen. Half say what they sell
# ("... MEDICALS"); half don't ("... ENTERPRISES"), and only your own filing
# can tell a model what those are.
LOCAL_NAMES = ["SRI VENKATESWARA", "SRI GANESH", "MURUGAN", "LAKSHMI", "SAI BABA", "BALAJI", "ANNAI",
               "KAMATCHI", "SELVI", "RAJA", "VINAYAGA", "AYYANAR", "MAHALAKSHMI", "SHANMUGA"]
LOCAL_KINDS = {
    "Food": (["TEA STALL", "MESS", "BIRYANI CENTRE", "BAKERY"], 120),
    "Groceries": (["PROVISION STORES", "VEGETABLES", "FRUITS STALL"], 200),
    "Health": (["MEDICALS", "CLINIC"], 250),
    "Transport": (["AUTO STAND", "TRAVELS"], 150),
    "Shopping": (["TEXTILES", "FOOTWEAR", "MOBILES"], 700),
    "Education": (["XEROX", "STATIONERY"], 60),
}
OPAQUE = ["ENTERPRISES", "TRADERS", "AGENCIES", "AND SONS", "STORE", "CORPORATION"]


@dataclass
class Txn:
    user: int
    on: date
    description: str
    amount: int          # paise, positive = money out
    merchant: str        # canonical merchant (first spelling)
    category: str
    sub_id: str | None = None     # set on every charge of a subscription
    cycle: str | None = None      # the subscription's cycle label
    anomaly: bool = False


@dataclass
class User:
    id: int
    rng: random.Random
    start: date
    days: int
    txns: list[Txn] = field(default_factory=list)


def _ref(rng: random.Random, n: int = 12) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


def narration(rng: random.Random, spelling: str, kind: str) -> str:
    """One bank narration for a payment to `spelling`, in a bank's style."""
    if kind == "upi":
        handle = spelling.lower().replace(" ", "")[:10] + "@" + rng.choice(HANDLES)
        return rng.choice([
            f"UPI/DR/{_ref(rng)}/{spelling}/{rng.choice(BANKS)}/{handle}",
            f"UPI-{spelling}-{handle}-{rng.choice(BANKS)}0{_ref(rng, 6)}-{_ref(rng)}-PAYMENT",
            f"UPI/{_ref(rng)}/Paid to {spelling}/{handle}",
        ])
    if kind == "pos":
        return rng.choice([
            f"POS {_ref(rng, 4)}XXXXXX{_ref(rng, 4)} {spelling} {rng.choice(CITIES)}",
            f"{spelling.replace(' ', '')}*{rng.choice(CITIES)[:3]} {_ref(rng, 4)}",
            f"ECOM PUR/{spelling}/{_ref(rng, 6)}",
        ])
    if kind == "mandate":
        return rng.choice([
            f"NACH/{spelling}/{_ref(rng, 10)}",
            f"SI {spelling} AUTOPAY {_ref(rng, 8)}",
            f"ACH D- {spelling}-{_ref(rng, 9)}",
            f"{spelling} RECURRING PMT {_ref(rng, 6)}",
        ])
    raise ValueError(kind)


def _amount(rng: random.Random, typical: float, spread: float = DEFAULT_SPREAD) -> int:
    rupees = typical * rng.lognormvariate(0, spread)
    rupees = round(rupees) if rng.random() < 0.7 else round(rupees, 2)
    return max(100, int(round(rupees * 100)))


def _add_months(d: date, n: int, day: int) -> date:
    m = d.month - 1 + n
    y, m = d.year + m // 12, m % 12 + 1
    for dd in (day, 30, 29, 28):
        try:
            return date(y, m, dd)
        except ValueError:
            continue
    raise AssertionError


def _subscriptions(u: User) -> None:
    rng = u.rng
    end = u.start + timedelta(days=u.days)
    for i, (spellings, category, price, cycle) in enumerate(rng.sample(SUBSCRIPTIONS, rng.randint(2, 6))):
        spelling = rng.choice(spellings)  # one spelling per user: banks are consistent
        sub_id = f"u{u.id}s{i}"
        begin = u.start + timedelta(days=rng.randint(0, u.days // 2) if rng.random() < 0.4 else rng.randint(0, 20))
        stop = begin + timedelta(days=rng.randint(60, u.days)) if rng.random() < 0.25 else end
        hike_on = begin + timedelta(days=rng.randint(60, u.days)) if rng.random() < 0.3 else None
        amount = int(price * 100)
        label = cycle if isinstance(cycle, str) else f"{cycle}d"
        k, on = 0, begin
        while on < min(stop, end):
            posted = on + timedelta(days=rng.choice([0, 0, 0, 1, 1, 2]))  # banks post late
            charge = amount if not hike_on or on < hike_on else int(round(amount * rng.choice([1.2, 1.25, 1.5]) / 100) * 100)
            if posted < end:
                u.txns.append(Txn(u.id, posted, narration(rng, spelling, rng.choice(["mandate", "mandate", "pos"])),
                                  charge, spellings[0], category, sub_id, label))
            k += 1
            on = (_add_months(begin, k, begin.day) if cycle == "month"
                  else _add_months(begin, 12 * k, begin.day) if cycle == "year"
                  else begin + timedelta(days=cycle * k))


def _rent(u: User) -> None:
    """Monthly rent / PG fee to a person: recurring, but not a 'subscription' brand."""
    if u.rng.random() > 0.5:
        return
    rng = u.rng
    who = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    amount = rng.choice([6000, 7500, 8000, 9500, 12000]) * 100
    day = rng.randint(1, 5)
    on = _add_months(u.start, 0, day)
    k = 0
    while on < u.start + timedelta(days=u.days):
        if on >= u.start:
            u.txns.append(Txn(u.id, on + timedelta(days=rng.choice([0, 0, 1, 2, 3])), narration(rng, who, "upi"),
                              amount, who, "Transfers", f"u{u.id}rent", "month"))
        k += 1
        on = _add_months(u.start, k, day)


def _everyday(u: User) -> None:
    rng = u.rng
    locals_ = _locals(rng)
    local_spread = {spellings[0]: rng.uniform(0.2, 0.5) for spellings, _, _ in locals_}
    favourites = rng.sample(EVERYDAY, rng.randint(12, 22)) + locals_
    weights = [rng.paretovariate(1.2) for _ in favourites]  # a few places get most of your money
    friends = [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(rng.randint(3, 8))]
    rate = rng.uniform(1.2, 3.5)  # payments per day
    for day in range(u.days):
        on = u.start + timedelta(days=day)
        busy = 1.4 if on.weekday() >= 5 else 1.0
        for _ in range(_poisson(rng, rate * busy)):
            if rng.random() < 0.12:  # splitting a bill with a friend
                who = rng.choice(friends)
                u.txns.append(Txn(u.id, on, narration(rng, who, "upi"), _amount(rng, 250, 0.6), who, "Transfers"))
                continue
            spellings, category, typical = rng.choices(favourites, weights)[0]
            spelling = rng.choice(spellings)
            kind = "upi" if rng.random() < 0.75 else "pos"
            spread = SPREAD.get(spellings[0], local_spread.get(spellings[0], DEFAULT_SPREAD))
            u.txns.append(Txn(u.id, on, narration(rng, spelling, kind), _amount(rng, typical, spread), spellings[0], category))


def _locals(rng: random.Random) -> list[tuple[list[str], str, float]]:
    shops = []
    for _ in range(rng.randint(4, 9)):
        category = rng.choice(list(LOCAL_KINDS))
        kinds, typical = LOCAL_KINDS[category]
        name = f"{rng.choice(LOCAL_NAMES)} {rng.choice(kinds if rng.random() < 0.5 else OPAQUE)}"
        shops.append(([name], category, typical))
    return shops


def _anomalies(u: User, rate: float = 0.01) -> None:
    """Plant unusual spends: a normal merchant, 4-10x its usual amount."""
    rng = u.rng
    normal = [t for t in u.txns if t.sub_id is None and t.category != "Transfers"]
    for t in rng.sample(normal, max(1, int(len(normal) * rate))):
        t.amount = int(t.amount * rng.uniform(4, 10))
        t.anomaly = True


def _poisson(rng: random.Random, lam: float) -> int:
    # Knuth; lam is small
    import math
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def generate(users: int = 40, days: int = 365, seed: int = 7, start: date = date(2025, 10, 1)) -> pd.DataFrame:
    """A statement per user, all in one frame, oldest first within each user."""
    rows: list[Txn] = []
    for uid in range(users):
        u = User(uid, random.Random(seed * 1000 + uid), start, days)
        _subscriptions(u)
        _rent(u)
        _everyday(u)
        _anomalies(u)
        rows += u.txns
    df = pd.DataFrame([t.__dict__ for t in rows])
    df["on"] = pd.to_datetime(df["on"])
    df = df.sort_values(["user", "on"], kind="stable").reset_index(drop=True)
    df.insert(0, "id", [f"t{i}" for i in range(len(df))])
    return df
