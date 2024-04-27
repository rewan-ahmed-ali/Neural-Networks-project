import numpy as np
import pandas as pd
import tensorflow as tf

# قراءة ملف البيانات
data = pd.read_csv("heart.csv")

# تقسيم البيانات إلى المتغيرات المستقلة (ميزات) والمتغير التابع (التصنيف)
X = data.drop(columns='target').values
y = data['target'].values


# تحويل التصنيفات إلى تنسيق فئوي واحد
num_classes = len(np.unique(y))
y = tf.keras.utils.to_categorical(y, num_classes)

# تقسيم البيانات إلى مجموعة التدريب ومجموعة الاختبار
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تحديد النموذج
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# تحديد المعيار ودالة الخسارة
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# تدريب النموذج
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# تقييم النموذج
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
