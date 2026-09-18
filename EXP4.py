import re
import pandas as pd
from urllib.parse import urlparse

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# 1. TRAINING DATASET
# =========================================================

legitimate_urls = [

    # Normal websites
    "https://www.google.com",
    "https://www.amazon.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.github.com",
    "https://stackoverflow.com",
    "https://www.wikipedia.org",
    "https://www.youtube.com",
    "https://www.linkedin.com",
    "https://www.netflix.com",
    "https://www.reddit.com",
    "https://www.python.org",
    "https://pypi.org",
    "https://www.nasa.gov",
    "https://www.cnn.com",
    "https://www.bbc.com",
    "https://www.coursera.org",
    "https://www.udemy.com",
    "https://www.flipkart.com",
    "https://www.myntra.com",

    # Legitimate URLs with parameters
    "https://www.google.com/search?q=machine+learning",
    "https://www.google.com/search?q=cybersecurity&hl=en",
    "https://www.amazon.com/s?k=wireless+headphones",
    "https://www.amazon.com/s?k=laptop&ref=nav_search",
    "https://github.com/search?q=cybersecurity&type=repositories",
    "https://stackoverflow.com/questions/tagged/python?sort=votes",
    "https://www.youtube.com/results?search_query=machine+learning",
    "https://www.reddit.com/search/?q=cybersecurity",
    "https://www.booking.com/searchresults.html?ss=London",
    "https://www.ebay.com/sch/i.html?_nkw=laptop",

    # Legitimate login/account pages
    "https://github.com/login",
    "https://www.paypal.com/signin",
    "https://accounts.google.com/signin",
    "https://login.microsoftonline.com/",
    "https://www.linkedin.com/login",
    "https://www.netflix.com/login",
    "https://support.google.com/accounts/",
    "https://www.apple.com/shop/buy-iphone/",
    "https://www.microsoft.com/en-us/security/",
    "https://www.amazon.com/ap/signin",

    # Legitimate but long/complex URLs
    "https://docs.python.org/3/library/urllib.parse.html",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://www.microsoft.com/en-us/security/business/security-101",
    "https://www.apple.com/in/shop/buy-iphone/iphone-16",
    "https://www.youtube.com/watch?v=abc123&feature=share",
    "https://github.com/search?q=machine+learning&type=code",
    "https://stackoverflow.com/questions/123456/example-question",
    "https://www.coursera.org/learn/machine-learning/lecture/example",
    "https://www.canva.com/templates/search/presentations/",
    "https://www.adobe.com/products/photoshop.html"
]


malicious_urls = [

    # IP address based URLs
    "http://192.168.1.10/login",
    "http://192.168.0.15/verify",
    "http://45.33.22.11/update",
    "http://103.44.55.66/login.php",
    "http://185.22.33.44/secure/login",

    # Fake PayPal
    "http://paypal-security-check.com",
    "http://paypal-login-verification.com",
    "http://secure-paypal-account.xyz",
    "http://paypal-account-confirmation.tk",
    "http://paypal-verify-login.ml",

    # Fake banking
    "http://bank-login-verify.com",
    "http://secure-bank-login.xyz",
    "http://online-banking-verification.tk",
    "http://bank-security-update.ml",
    "http://bank-account-confirmation.xyz",

    # Account phishing
    "http://verify-account-now.xyz",
    "http://confirm-your-account.tk",
    "http://account-verification-required.ml",
    "http://security-alert-login.xyz",
    "http://your-account-is-locked.tk",

    # Password phishing
    "http://update-your-password.ml",
    "http://password-reset-confirm.tk",
    "http://login-confirm-account.tk",
    "http://signin-verification-required.xyz",
    "http://account-password-update.xyz",

    # Fake rewards
    "http://free-gift-card.xyz",
    "http://you-have-won-prize.tk",
    "http://claim-free-reward.ml",
    "http://iphone-free-giveaway.xyz",
    "http://free-money-reward.tk",

    # URLs using @ to disguise destination
    "http://google.com@malicious-site.com/login",
    "http://paypal.com@evil-example.com",
    "http://facebook.com@account-verification.xyz",
    "http://microsoft.com@login-security.tk",
    "http://amazon.com@fake-login.xyz",

    # Suspicious long URLs
    "http://secure-login-example.com/verify/account/login.php?session=123456789",
    "http://account-security-update.com/login.php?user=admin&password=verify",
    "http://payment-confirmation.xyz/secure/login/account/verify.php?id=12345",
    "http://online-banking-security.tk/update/password/confirm/login.php",
    "http://verify-account-now.xyz/login.php?session=987654&token=abcdef",

    # URL shorteners
    "https://bit.ly/3abcXYZ",
    "https://tinyurl.com/login-update",
    "http://bit.ly/free-gift-card",
    "https://t.co/account-verification",
    "https://is.gd/security-login",

    # Suspicious subdomains
    "http://login.security.verify.example-malicious.com",
    "http://secure.account.update.example-phishing.com",
    "http://paypal.login.verify.evil-example.net",
    "http://bank.security.confirm.fake-domain.org",
    "http://account.verify.security.fake-site.com",

    # Fake downloads
    "http://download-free-software.xyz/setup.exe",
    "http://free-movie-download.tk/movie.exe",
    "http://crack-software-download.ml/keygen.exe",
    "http://urgent-security-update.xyz/update.exe",
    "http://free-antivirus-download.tk/install.exe",

    # Urgency based phishing
    "http://urgent-security-message.xyz",
    "http://unusual-login-detected.xyz/verify",
    "http://important-security-message.tk",
    "http://verify-payment-information.xyz",
    "http://confirm-credit-card-details.tk"
]


# =========================================================
# 2. AUTOMATICALLY CREATE LABELS
# =========================================================

urls = legitimate_urls + malicious_urls

labels = (
    [0] * len(legitimate_urls) +
    [1] * len(malicious_urls)
)

df = pd.DataFrame({
    "url": urls,
    "label": labels
})

print("Total URLs:", len(df))
print("Legitimate URLs:", len(legitimate_urls))
print("Malicious URLs:", len(malicious_urls))


# =========================================================
# 3. FEATURE EXTRACTION
# =========================================================

def extract_features(url):

    # Add protocol if missing
    if not url.startswith(("http://", "https://")):
        url_to_parse = "http://" + url
    else:
        url_to_parse = url

    parsed = urlparse(url_to_parse)

    domain = parsed.netloc
    path = parsed.path
    query = parsed.query

    # Remove username information
    clean_domain = domain.split("@")[-1]
    clean_domain = clean_domain.split(":")[0]

    # 1. URL length
    url_length = len(url)

    # 2. Domain length
    domain_length = len(clean_domain)

    # 3. Number of dots
    dots = url.count(".")

    # 4. Number of hyphens
    hyphens = url.count("-")

    # 5. Number of digits
    digits = sum(c.isdigit() for c in url)

    # 6. Number of special characters
    special_characters = sum(
        not c.isalnum() for c in url
    )

    # 7. Number of subdomains
    domain_parts = clean_domain.split(".")

    if len(domain_parts) > 2:
        subdomains = len(domain_parts) - 2
    else:
        subdomains = 0

    # 8. Check for IP address
    ip_pattern = r"^\d{1,3}(\.\d{1,3}){3}$"

    has_ip = int(
        bool(re.match(ip_pattern, clean_domain))
    )

    # 9. @ symbol
    has_at_symbol = int("@" in url)

    # 10. HTTPS
    has_https = int(
        url.lower().startswith("https://")
    )

    # 11. Suspicious words
    suspicious_words = [
        "login",
        "signin",
        "verify",
        "verification",
        "update",
        "secure",
        "security",
        "account",
        "confirm",
        "password",
        "bank",
        "payment",
        "free",
        "reward",
        "prize",
        "urgent",
        "locked",
        "credential"
    ]

    suspicious_word_count = sum(
        word in url.lower()
        for word in suspicious_words
    )

    # 12. URL shortener
    shorteners = [
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "goo.gl",
        "ow.ly",
        "is.gd",
        "buff.ly"
    ]

    has_shortener = int(
        any(
            shortener in domain.lower()
            for shortener in shorteners
        )
    )

    # 13. Query length
    query_length = len(query)

    # 14. Path length
    path_length = len(path)

    return [
        url_length,
        domain_length,
        dots,
        hyphens,
        digits,
        special_characters,
        subdomains,
        has_ip,
        has_at_symbol,
        has_https,
        suspicious_word_count,
        has_shortener,
        query_length,
        path_length
    ]


# =========================================================
# 4. CREATE FEATURE MATRIX
# =========================================================

feature_names = [
    "URL_Length",
    "Domain_Length",
    "Dots",
    "Hyphens",
    "Digits",
    "Special_Characters",
    "Subdomains",
    "Has_IP",
    "Has_At_Symbol",
    "Has_HTTPS",
    "Suspicious_Words",
    "Has_URL_Shortener",
    "Query_Length",
    "Path_Length"
]


X = pd.DataFrame(
    [
        extract_features(url)
        for url in df["url"]
    ],
    columns=feature_names
)

y = df["label"]


# =========================================================
# 5. SPLIT DATASET
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# =========================================================
# 6. TRAIN RANDOM FOREST
# =========================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)


# =========================================================
# 7. EVALUATE MODEL
# =========================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Legitimate",
            "Malicious"
        ]
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# =========================================================
# 8. USER-DRIVEN CLASSIFICATION
# =========================================================

print("\n========================================")
print("   MALICIOUS URL DETECTOR")
print("========================================")

print("\nEnter a URL to classify.")
print("Type 'exit' to stop.\n")


while True:

    url = input("Enter URL: ").strip()

    # Exit
    if url.lower() == "exit":
        print("\nProgram terminated.")
        break

    # Empty input
    if not url:
        print("Please enter a URL.\n")
        continue

    # Extract features
    url_features = extract_features(url)

    # Convert to DataFrame
    input_data = pd.DataFrame(
        [url_features],
        columns=feature_names
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probabilities = model.predict_proba(
        input_data
    )[0]

    legitimate_probability = probabilities[0] * 100
    malicious_probability = probabilities[1] * 100

    print("\n----------------------------------------")

    print("URL:", url)

    print(
        "Legitimate Probability:",
        round(legitimate_probability, 2),
        "%"
    )

    print(
        "Malicious Probability:",
        round(malicious_probability, 2),
        "%"
    )

    if prediction == 1:

        print(
            "Classification: "
            "MALICIOUS / SUSPICIOUS"
        )

    else:

        print(
            "Classification: "
            "LEGITIMATE"
        )

    print("----------------------------------------\n")
