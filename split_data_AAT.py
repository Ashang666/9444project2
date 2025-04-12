import pandas as pd
from sklearn.model_selection import train_test_split

# 加载 AAT 标签文件
df = pd.read_csv('image_labels_aat.csv')

# 按照子类标签 stratify 分层划分，保持类别分布一致
train_val, test = train_test_split(df, test_size=0.15, stratify=df['label_sub'], random_state=42)
train, val = train_test_split(train_val, test_size=0.15, stratify=train_val['label_sub'], random_state=42)

# 保存
train.to_csv('train_b.csv', index=False)
val.to_csv('val_b.csv', index=False)
test.to_csv('test_b.csv', index=False)

print(f"✅ 数据划分完成：Train {len(train)}, Val {len(val)}, Test {len(test)}")
