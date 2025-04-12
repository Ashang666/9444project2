import json
import csv

# 设置文件路径（你可以换成你自己的路径）
aat_json_path = '/Users/liujiaxiang/Desktop/9444gs/【AiDLab】A100/AAT/label/AAT.json'
output_csv = "image_labels_aat.csv"

image_labels = []

with open(aat_json_path, "r") as f:
    data = json.load(f)  # data 是一个 list

    for item in data:
        # 提取答案中的每个 image 名称（格式：类别/子类_编号）
        for ans in item['answers']:
            image_name = ans.split("_")[1] + ".jpg"  # e.g., Shoes/Heels_001A03 → 001A03.jpg
            class_path = ans.split("_")[0]           # e.g., Shoes/Heels
            if "/" in class_path:
                main_class, sub_class = class_path.split("/", 1)
            else:
                main_class, sub_class = class_path, ""
            image_labels.append([image_name, main_class, sub_class])

# 写入 CSV 文件
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "label_main", "label_sub"])
    writer.writerows(image_labels)

print(f"✅ 已生成标签文件，共 {len(image_labels)} 条记录：{output_csv}")
