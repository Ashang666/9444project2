import json
import pandas as pd
import os

# TODO: 修改为你的 LAT.json 文件路径
json_path = '/Users/mashang/Desktop/9444as2/【AiDLab】A100/LAT/label/LAT.json'

# 定义细类到主类的映射
def get_main_category(sub_label):
    if sub_label in ['Pants', 'Skirt', 'Top', 'Outwear']:
        return 'Clothing'
    elif sub_label == 'Shoes':
        return 'Shoes'
    elif sub_label == 'Bags':
        return 'Bags'
    elif sub_label in ['Earing', 'Bracelet', 'Watches', 'Necklace']:
        return 'Accessories'
    else:
        return 'Unknown'

# 加载 JSON 数据
with open(json_path, 'r') as f:
    data = json.load(f)

# 构建图片到标签的映射
img_label_map = {}

for entry in data:
    for group in ['question', 'answers']:
        for item in entry[group]:
            if '_' in item:
                subcat, img_id = item.split('_')
                main_cat = get_main_category(subcat)
                if main_cat != 'Unknown':
                    img_label_map[img_id + '.jpg'] = main_cat

# 转换为 DataFrame 保存为 CSV 文件
df = pd.DataFrame(list(img_label_map.items()), columns=['image', 'label'])

# 输出 CSV 到项目根目录
df.to_csv('image_labels.csv', index=False)

print(f"✅ 已成功生成 image_labels.csv 文件，共 {len(df)} 条图像标签。")
