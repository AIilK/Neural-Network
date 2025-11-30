# --- مرحله صفر: وارد کردن کتابخانه‌ها ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- مرحله 1: خواندن داده‌ها ---
zoo_df = pd.read_csv('zoo.csv')
class_df = pd.read_csv('class.csv')

# --- مرحله 2: ادغام داده‌ها ---
merged_df = pd.merge(zoo_df,class_df,how='left',left_on='class_type',right_on='Class_Number')


# نمایش چند ردیف اول برای بررسی
print("part of test")
print(merged_df.head())

# رسم نمودار توزیع کلاس‌ها
plt.figure(figsize=(8, 4))
sns.countplot(data=merged_df, x='Class_Type')
plt.title('tozi data')
plt.xlabel('class')
plt.ylabel('num')
plt.show()

# بررسی تعادل کلاس‌ها
print("\tedad nemone ")
print(merged_df['Class_Type'].value_counts())

# --- مرحله 3: تقسیم داده‌ها به آموزش و آزمون ---
# حذف ستون‌هایی که برای آموزش مناسب نیستند
X = merged_df.drop(columns=['animal_name', 'Class_Type'])
y = merged_df['Class_Type']

# تقسیم‌بندی
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

# --- مرحله 4: آموزش مدل پرسپترون چند لایه ---
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# پیش‌بینی و ارزیابی مدل
y_pred = model.predict(X_test)

print("\n accury", accuracy_score(y_test, y_pred))
print("\nreport classification")
print(classification_report(y_test, y_pred))
