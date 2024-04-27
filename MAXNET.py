import numpy as np
import pandas as pd

# قراءة ملف البيانات
data = pd.read_csv("heart.csv")

# تحديد البيانات
X = data.drop(columns='target').values

# تحديد الوزن الثابتة
w = np.eye(X.shape[1])

# تطبيق الشبكة MaxNET
output = np.dot(X, w.T)

print("Output:", output)
