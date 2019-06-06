import pandas as pd
import numpy as np
import matplotlib
from  chapter09.ModelTree import *
from tkinter import *
#导入渲染器，功能是执行绘画等动作
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#导入画布
from matplotlib.figure import Figure

def reDraw(tolS,tolN):
    root = Tk()  # 实例化
    myLabel = Label(root, text='hello world')  # 设置Label部件
    myLabel.grid()  # 布局管理器
    reDraw.f = Figure(figsize = (5,4),dpi= 100)   #创建画布
    reDraw.f.clf()  #清空画布
    reDraw.canvas = FigureCanvasTkAgg(reDraw.f,master = root) #初始化渲染器
    reDraw.canvas.get_tk_widget().grid(row=0,columnspan=3) #返回用于实现FigureCanvasTkAgg的Tk小部件
    reDraw.a = reDraw.f.add_subplot(111) #创建子画布
    if chkBtnVar.get():
        if tolN<2: tolN=2  #我们使用的数据集tolN<2时，矩阵是奇异矩阵
        myTree = createTree(reDraw.rawDat,modelLeaf,modelError,ops = (tolS,tolN))
        yHat = createForeCast(myTree,reDraw.testDat,modelTreeEval)
    else:
        myTree = createTree(reDraw.rawDat,ops=(tolS,tolN))
        yHat = createForeCast(myTree,reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat.iloc[:,0],reDraw.rawDat.iloc[:,1],s=5,c='darkorange')
    reDraw.a.plot(reDraw.testDat.iloc[:,0],yHat,linewidth=2.0,c = 'yellowgreen')
    reDraw.canvas.draw()

def getInputs():
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print('请输入浮点数')
        tolSentry.delete(0,END)
        tolSentry.insert(0,'1.0')
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print('请输入整数')
        tolNentry.delete(0,END)
        tolNentry.insert(0,'10')
    return tolS,tolN

#利用用户输入值绘制树
def drawNewTree():
    tolS,tolN = getInputs()
    reDraw(tolS,tolN)

#实例化一个窗口对象
root = Tk()
#窗口的标题
root.title('回归树调参')
#tolS
Label(root,text = 'tolS').grid(row = 1, column = 0)
tolSentry = Entry(root) #entry：单行文本输入框
tolSentry.grid(row=1, column = 1)
tolSentry.insert(0,'1.0') #默认值为1.0
#tolN
Label(root,text = 'tolN').grid(row = 2, column = 0)
tolNentry = Entry(root) #Entry：单行文本输入框
tolNentry.grid(row=2, column = 1)
tolNentry.insert(0,'10') #默认值为10
#按钮
Button(root,text = 'ReDraw',command = drawNewTree).grid(row = 1, column = 2,rowspan =3)
#按钮整数值
chkBtnVar = IntVar()
#复选按钮
chkBtn = Checkbutton(root,text = 'Model Tree',variable = chkBtnVar)
chkBtn.grid(row = 3, column = 0,columnspan = 2)
#导入数据
dataSet = pd.read_table("C:\\Users\\Mypc\\Desktop\\第8期 树回归（完整版）\\sine.txt",header=None)
train = dataSet[:160]
test = dataSet[160:].sort_values(by=dataSet[160:].columns[0]) #按特征从小到大排序
test.index = range(test.shape[0]) #更新索引
reDraw.rawDat = train
reDraw.testDat = test

reDraw(1.0,10)

root.mainloop()