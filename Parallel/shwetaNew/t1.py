mydict = {'num_NNlayers':'', 'num_features':'', 'hidden_nodes':'', 'num_labels':'', 'learning_rate':'', 'steps':'', 'batch_size':''}
def getParas(fileName):
    dictNew = {}
    with open(fileName, 'r') as f:
        for line in f:
            for i in mydict.keys():
                if i in line:
                    print i
                    dictNew[i] = line.split(':')[1].split('\n')[0]
                    break;
    f.close()

    for i in dictNew.keys():
        if i is "learning_rate":
            dictNew[i] = float(dictNew[i])
        else:
            dictNew[i] = int(dictNew[i])
    print dictNew
    return dictNew


paras = getParas("para.txt")


print(paras['num_NNlayers'])