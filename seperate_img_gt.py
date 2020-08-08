import os,shutil
#输入数据的路径，在该数据的路径下建立img和gt文件夹，分别存放img文件和gt文件


def remove_file(data_path):

    txt_file = [os.path.join(data_path,path) for path in os.listdir(data_path) if path.endswith(".txt")]
    img_file = [ txt.replace(".txt", ".jpg") for txt in txt_file]

    new_txt_dir = os.path.join(os.path.dirname(data_path), "gt")
    new_img_dir = os.path.join(os.path.dirname(data_path), "img")

    if os.path.exists(new_txt_dir):
        print("already exist txt_dir")
    else:
        os.mkdir(new_txt_dir)

    if os.path.exists(new_img_dir):
        print("already exist img_dir")
    else:
        os.mkdir(new_img_dir)


    for txt in txt_file:
        print(txt)
        base_dir = os.path.basename(txt)
        shutil.copy(txt, os.path.join(new_txt_dir, base_dir))

    for img in img_file:
        base_dir = os.path.basename(img)
        shutil.copy(img, os.path.join(new_img_dir, base_dir))

if __name__ == "__main__":
    data_path = r"F:\training_data\normal_data"
    remove_file(data_path)

