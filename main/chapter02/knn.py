from os import listdir
from chapter02.numpyTest import *
def img2vector(filename):
    returnVect =zeros((1,1024))
    fr= open(filename)
    for i in range(32):
        lineStr =fr.readline()
        for j in range(32):
            returnVect[0 ,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLables= []
    fileList = listdir("C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch02\\digits\\trainingDigits")
    m = len(fileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = fileList[i]
        fileName = fileNameStr.split('.')[0]
        fileLable =fileName.split('_')[0]
        hwLables.append(int(fileLable))
        returnVect = img2vector('C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch02\\digits\\trainingDigits\\'+fileNameStr)
        trainingMat[i,:] = returnVect
    testFileList = listdir("C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch02\\digits\\testDigits")
    n = len(testFileList)
    errorCount= 0.0
    for j in range(n):
        fileNameStr = fileList[j]
        fileName = fileNameStr.split('.')[0]
        fileLable = fileName.split('_')[0]
        returnVect = img2vector('C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch02\\digits\\trainingDigits\\'+fileNameStr);
        result = classify0(returnVect,trainingMat,hwLables,3)
        print("the calssifier came back with: %d,the real answer is:%d",result , fileLable)
        if(result != fileLable):
            errorCount += 1
    print("the total error rate is: %f" ,(errorCount / float(n)) ) # 最后打印出测试错误率)

# returnVect = img2vector("C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch02\\digits\\trainingDigits\\0_1.txt")
# print(returnVect)
# handwritingClassTest()