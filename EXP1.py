import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate sample dates
dates = pd.date_range(start="2024-01-01", periods=120, freq="D")

# Create trend, seasonal, and noise components
trend = np.linspace(1000, 1800, 120)
seasonal = 200 * np.sin(np.arange(120) * 2 * np.pi / 7)
noise = np.random.normal(0, 50, 120)

# Combine all components to create network traffic data
traffic = trend + seasonal + noise

# Create DataFrame
data = pd.DataFrame({
    "Date": dates,
    "Network_Traffic": traffic
})

# Set Date as index
data.set_index("Date", inplace=True)

# Perform time series decomposition
result = seasonal_decompose(
    data["Network_Traffic"],
    model="additive",
    period=7
)

# Plot decomposition
result.plot()
plt.suptitle("Time Series Decomposition of Network Traffic", fontsize=14)
plt.show()
