with open("replacements.txt", "r") as f:
    with open("replacements_second.txt", "r") as f2:
        with open("replacements_combined.txt", "w") as f3:
            lines = f.readlines()
            lines2 = f2.readlines()
            for line in lines:
                f3.write(line)
            for line in lines2:
                f3.write(line)

