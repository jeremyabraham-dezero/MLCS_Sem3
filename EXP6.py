{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "37060d54-9d3e-4a24-b3e2-9af5f332366d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "URL Anomaly Detection\n",
      "\n",
      "Normal    : https://google.com\n",
      "Normal    : https://youtube.com\n",
      "Normal    : https://github.com\n",
      "Normal    : https://amazon.com\n",
      "Normal    : https://example.com\n",
      "Normal    : http://test.com\n",
      "Normal    : http://192.168.1.1/login\n",
      "Normal    : http://abc.com/login\n",
      "Anomaly   : http://free-money-win-prize.com/offer\n",
      "Anomaly   : http://x.com/a/b/c/d/e/f/g/h/i/j\n"
     ]
    }
   ],
   "source": [
    "# URL Anomaly Detection using Feature Engineering\n",
    "\n",
    "import pandas as pd\n",
    "from sklearn.ensemble import IsolationForest\n",
    "\n",
    "# Sample URLs\n",
    "urls = [\n",
    "    \"https://google.com\",\n",
    "    \"https://youtube.com\",\n",
    "    \"https://github.com\",\n",
    "    \"https://amazon.com\",\n",
    "    \"https://example.com\",\n",
    "    \"http://test.com\",\n",
    "    \"http://192.168.1.1/login\",\n",
    "    \"http://abc.com/login\",\n",
    "    \"http://free-money-win-prize.com/offer\",\n",
    "    \"http://x.com/a/b/c/d/e/f/g/h/i/j\"\n",
    "]\n",
    "\n",
    "# Feature extraction\n",
    "def extract_features(url):\n",
    "    return [\n",
    "        len(url),                 # URL length\n",
    "        url.count(\".\"),           # Number of dots\n",
    "        url.count(\"/\"),           # Number of slashes\n",
    "        url.count(\"-\"),           # Number of hyphens\n",
    "        url.count(\"?\")            # Number of ?\n",
    "    ]\n",
    "\n",
    "# Create feature dataset\n",
    "features = [extract_features(url) for url in urls]\n",
    "\n",
    "X = pd.DataFrame(\n",
    "    features,\n",
    "    columns=[\"Length\", \"Dots\", \"Slashes\", \"Hyphens\", \"QuestionMarks\"]\n",
    ")\n",
    "\n",
    "# Create Isolation Forest model\n",
    "model = IsolationForest(\n",
    "    contamination=0.2,\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "# Train model\n",
    "model.fit(X)\n",
    "\n",
    "# Predict anomalies\n",
    "prediction = model.predict(X)\n",
    "\n",
    "# Display results\n",
    "print(\"URL Anomaly Detection\\n\")\n",
    "\n",
    "for url, result in zip(urls, prediction):\n",
    "\n",
    "    if result == -1:\n",
    "        print(\"Anomaly   :\", url)\n",
    "    else:\n",
    "        print(\"Normal    :\", url)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7b326dac-fcc4-4a19-86e7-a387aed3dcbd",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
