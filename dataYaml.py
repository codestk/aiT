import os

def create_data_yaml(path_to_classes_txt, path_to_data_yaml):
    # 1) อ่านคลาส
    if not os.path.exists(path_to_classes_txt):
        print(f'classes.txt file not found! Please create a classes.txt at {path_to_classes_txt}')
        return
    with open(path_to_classes_txt, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]

    # 2) สร้าง path ให้ชัวร์ด้วย os.path.join (กันปัญหา \t, \v)
    base = r'D:\aiT\data'
    train_images = os.path.join(base, 'train', 'images')
    val_images   = os.path.join(base, 'validation', 'images')

    # 3) ทำ inline list เอง
    names_inline = "[" + ", ".join(f'"{c}"' for c in classes) + "]"

    # 4) ประกอบเป็นข้อความ YAML ตรงรูปแบบที่ต้องการ
    content = (
        f"path: {base}\n"
        f"train: {train_images}\n"
        f"val: {val_images}\n"
        f"nc: {len(classes)}\n"
        f"names: {names_inline}\n"
    )

    # 5) เขียนไฟล์
    with open(path_to_data_yaml, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f'Created config file at {path_to_data_yaml}')


# Define path to classes.txt and run function
path_to_classes_txt = 'custom_data/classes.txt'
path_to_data_yaml = 'data.yaml'

create_data_yaml(path_to_classes_txt, path_to_data_yaml)

print('\nFile contents:\n')
# cat /content/data.yaml