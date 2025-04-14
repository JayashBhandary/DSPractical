import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = pd.read_csv(r"wholesale.csv")
categorical_features = ['Channel', 'Region']
continuous_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
print(data[continuous_features].describe())
for col in categorical_features:
    dummies = pd.get_dummies(data[col], prefix=col)
    data = pd.concat([data, dummies], axis=1)
    data.drop(col, axis=1, inplace=True)
mms = MinMaxScaler()
data_transformed = mms.fit_transform(data)
sum_of_squared_distances = []
K = range(1, min(10, len(data_transformed)) + 1)  # Limit K to number of samples
for k in K:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)  # Explicitly set n_init
    km.fit(data_transformed)
    sum_of_squared_distances.append(km.inertia_)
# Plot the Elbow Method graph
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method for Optimal k')
plt.show()
