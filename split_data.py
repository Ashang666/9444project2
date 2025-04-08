import pandas as pd
from sklearn.model_selection import train_test_split

# 读取原始标签文件
df = pd.read_csv('image_labels.csv')

# 先划分出训练集（70%）
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)

# 再将剩下的 30% 平分为验证集和测试集（各 15%）
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

# 保存成三个 csv 文件
train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)
test_df.to_csv('test.csv', index=False)

print(f'✅ 数据集划分完成！\n训练集: {len(train_df)} 条\n验证集: {len(val_df)} 条\n测试集: {len(test_df)} 条')
