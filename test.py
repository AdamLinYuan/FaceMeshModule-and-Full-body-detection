array = []
count = 0
for s in f:
    if int(s) > 0:
        count += 1
    elif int(s) == 0:
        array.append(count)
        count = 0
print(max(array))