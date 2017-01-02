import mfi
import tools
import numpy as np
import pickle
import pdb
import os
import view
from sklearn.svm import SVC

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.svm.libsvm import decision_function




# CREATE RESULT FOLDER
if not os.path.exists("results"):
    os.makedirs("results")


# READ DATA
xtrain,ytrain,xtest,ytest = tools.read_data()
numbers = [4,9]
x,y = tools.choose_numbers(xtrain,ytrain,numbers)
xt,yt = tools.choose_numbers(xtest,ytest,numbers)




# TRAIN SVM
clf = SVC()
clf.fit(x, y)

# GENERATE RANDOM SAMPLES
samplesize = 5000
samples = np.random.uniform(-1.,1.,(samplesize,len(x[0])))#np.random.uniform(0.,1.,(samplesize,len(x[0])))


# INSTANCE-BASED MFI

C = np.array([0,5,77,113])
N=len(C)
ibr = []

'''
compute = False

if compute:
    for i in range(N):
        ibr.append(np.sign(clf.decision_function(x[C[i]]))*tools.mfi_ib(clf, 'decision_function', samples, x[C[i]]))
    fobj = open('test.pkl','wb')
    pickle.dump(ibr,fobj)
    fobj.close()
else:
    fobj = open('test.pkl','rb')
    ibr = pickle.load(fobj)
    fobj.close()


# PIXEL FLIPPING
flibos = []
for i in range(len(ibr)):
    flibos.append(tools.flibo_relation_rep(clf,x[C[i]],ibr[i],N=100,rep=5))

print flibos


view.PLOTnum3(ibr,"results/mfi_ibr.png",x[C],clf.decision_function(x[C]))

'''



# GENERATE RANDOM SAMPLES
samplesize = 1000
samples = np.random.uniform(-1.,1.,(samplesize,len(x[0])))#np.random.uniform(0.,1.,(samplesize,len(x[0])))
metric = 'rbf'

# MODEL-BASED MFI
mbr = tools.mfi_mb(clf.decision_function(samples), samples, metric, degree=2)
#view.PLOTnum(mbr,"results/mfi_mbr.pdf",np.reshape(np.mean(x,axis=0),(16,16)))
view.PLOTnum3([mbr],"results/mfi_mbr.png",[np.mean(x[y==numbers[0]],axis=0)])

pdb.set_trace()
view.PLOT(mbr,"results/mfi_mbr.png")

py.subplot(2, len(mfis), len(mfis) + 1 + i)
fig = py.imshow(rgb)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
py.title(str(np.round(prediction[i],2)))

py.savefig(savepath, bbox_inches=0, orientation='landscape', pad_inches=0.1)
