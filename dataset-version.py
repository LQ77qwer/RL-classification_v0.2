import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('new_merged_dataset.csv')
pd.set_option('display.max_columns', None)  # None意味着不限制显示的列数
pd.set_option('display.max_rows', None)  # 如果你也想显示所有行，可以设置这个选项
# print(df['cbo'].describe())


'''低相关性查找'''
# Calculate the correlation matrix
correlation_matrix = df.corr()

# Extract the correlation values with the target variable 'bug'
feature_correlation_with_bug = correlation_matrix['bug'].sort_values()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Generate a bar plot
sns.barplot(x=feature_correlation_with_bug.values, y=feature_correlation_with_bug.index, palette="viridis")

# Add labels and title
plt.title('Correlation of Features with the Target Variable "bug"')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')

# Show the plot
# plt.tight_layout()
# plt.show()


'''随机森林特征重要性评估'''
X = df.drop('bug', axis=1)
y = df['bug']
# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# 获取特征重要性
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)

print(feature_importances)
