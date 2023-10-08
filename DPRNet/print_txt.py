


f = open("./data/train/val-6000.txt", 'w')
for i in range(2801, 3001):
    f.write("train-6000/hazy/" + str('%.4d'%i) + ".jpg\n")
for i in range(5801, 6001):
    f.write("train-6000/hazy/" + str('%.4d'%i) + ".jpg\n")
f.close()