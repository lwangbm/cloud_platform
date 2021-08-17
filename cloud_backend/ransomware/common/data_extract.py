file_in_name = "../data/0812/training/DATAPOSITIVE/200hz_normal20210812_101154.txt"
file_out_name = "../data/0812/training/DATAPOSITIVE/200hz_normal20210812_101155.txt"
index = 0
max_index = 700000
with open(file_in_name, "r", encoding="UTF-8") as file_in:
    with open(file_out_name, "w") as file_out:
        for item in file_in.read().split("\n"):
            file_out.write("%s\n" % item)
            index += 1
            if index >= max_index:
                break
