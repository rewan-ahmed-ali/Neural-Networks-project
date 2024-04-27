import numpy as np
import pandas as pd
import tensorflow as tf

# قراءة ملف البيانات
data = pd.read_csv("heart.csv")

# تقسيم البيانات إلى المتغيرات المستقلة (ميزات) والمتغير التابع (التصنيف)
X = data.drop(columns='target').values
y = data['target'].values

# تحديد النموذج
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # استخدام sigmoid للتنبؤ بقيمة ثنائية (0 أو 1)
])

# تحديد المعيار ودالة الخسارة
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# تدريب النموذج
model.fit(X, y, epochs=50, batch_size=32)

# تقييم النموذج (اختياري إذا كان لديك بيانات اختبار)
loss, accuracy = model.evaluate(X, y)
print("Test Accuracy:", accuracy)

# استخدام النموذج للتنبؤ بالنتائج
# بما أن النموذج يستخدم sigmoid، فإن القيمة الأكبر من 0.5 تعني التنبؤ بالإيجابية (مرض القلب)، والأقل تعني التنبؤ بالسلبية (عدم وجود مرض القلب)
input_data = np.array([[54, 1, 0, 120, 188, 0, 1, 113, 0, 1.4, 1, 1, 3]])
prediction = model.predict(input_data)
if prediction[0][0] >= 0.5:
    print("The person is predicted to have heart disease.")
else:
    print("The person is predicted not to have heart disease.")
