import pickle
data = pickle.load(open("index.pickle","rb"))
output = open("index1.txt","w")
output.write(str(data))
output.flush()
output.close()