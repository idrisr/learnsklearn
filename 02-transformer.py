from sklearn.preprocessing import StandardScaler
X = [[0, 15], [1, -10]]
scaler = StandardScaler()
s = scaler.fit(X).transform(X)
