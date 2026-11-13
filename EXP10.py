
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Create sample network dataset
data = {
    "packet_count": [100, 120, 110, 130, 115,
                     900, 1000, 950, 80, 105],
    "duration": [10, 12, 11, 13, 10,
                 2, 1, 3, 9, 11],
    "failed_logins": [0, 0, 1, 0, 0,
                      20, 25, 18, 0, 1]
}

df = pd.DataFrame(data)

# Select features
X = df[["packet_count", "duration", "failed_logins"]]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
model = KMeans(n_clusters=2, random_state=42, n_init=10)
df["cluster"] = model.fit_predict(X_scaled)

# Identify potentially abnormal cluster
suspicious_cluster = (
    df.groupby("cluster")["failed_logins"].mean().idxmax()
)

# Display results
for i, row in df.iterrows():

    if row["cluster"] == suspicious_cluster:
        status = "Abnormal"
    else:
        status = "Normal"

    print("Event", i + 1, "-", status)
Event 1 - Normal
Event 2 - Normal
Event 3 - Normal
Event 4 - Normal
Event 5 - Normal
Event 6 - Abnormal
Event 7 - Abnormal
Event 8 - Abnormal
Event 9 - Normal
Event 10 - Normal
C:\Users\ADDWIN\AppData\Roaming\Python\Python313\site-packages\sklearn\cluster\_kmeans.py:1425: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
  warnings.warn(
 
