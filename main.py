from river import stream


X_y = stream.iter_csv("data/cleaned_speeddating.csv")
x, y = next(X_y)
t