import sklearn
import pandas
import time

from sklearn import datasets
df = pandas.read_csv('test_set.csv')
dataset = df.values
labels=dataset[:,4]
data=dataset[:,5:20]

#test file
df_test = pandas.read_csv('new_data.csv')
dataset_test = df_test.values
labels_test=dataset_test[:,4]
data_test=dataset_test[:,5:20]

#classifier
begin_time=time.time()
from sklearn import tree#DT
cls1 = tree.DecisionTreeClassifier(criterion='gini',max_depth=10,splitter='best',min_samples_split=2)
cls1=cls1.fit(data,labels)
score1=cls1.score(data_test,labels_test)
end_time=time.time()
time1=end_time-begin_time
print("DT\n")
print("Running time for this classifer:",time1,"s")
print("The score of this model is: %f"%score1)
print("\n")
print("\n")

begin_time=time.time()
from sklearn.svm import SVC#SVM
cls2 = SVC(kernel='rbf', gamma=0.1)
cls2=cls2.fit(data,labels)
score2=cls2.score(data_test,labels_test)
end_time=time.time()
time2=end_time-begin_time
print("SVM\n")
print("Running time for this classifer:",time2,"s")
print("The score of this model is: %f"%score2)
print("\n")
print("\n")

begin_time=time.time()
from sklearn.neighbors import KNeighborsClassifier#KNN
cls3 = KNeighborsClassifier(n_neighbors=5)
cls3=cls3.fit(data,labels)
score3=cls3.score(data_test,labels_test)
end_time=time.time()
time3=end_time-begin_time
print("KNN\n")
print("Running time for this classifer:",time3,"s")
print("The score of this model is: %f"%score3)
print("\n")
print("\n")

begin_time=time.time()
from sklearn.neural_network import MLPClassifier#MLP
cls4 = MLPClassifier(max_iter=10) 
cls4=cls4.fit(data,labels)
score4=cls4.score(data_test,labels_test)
end_time=time.time()
time4=end_time-begin_time
print("MLP\n")
print("Running time for this classifer:",time4,"s")
print("The score of this model is: %f"%score4)
print("\n")
print("\n")

#ensemble
begin_time=time.time()
from sklearn.ensemble import VotingClassifier#voting classifier
voting1 = VotingClassifier(estimators=[          #hard
    ('DT_clf',tree.DecisionTreeClassifier()),
    ('SVM_clf',SVC(kernel='rbf', gamma=0.1)),
    ('KNN_clf',KNeighborsClassifier(n_neighbors=5)),
    ('MLP_clf',MLPClassifier(max_iter=10))
    ], voting='hard')
voting1=voting1.fit(data,labels)
score5=voting1.score(data_test,labels_test)
end_time=time.time()
time5=end_time-begin_time
print("Ensemble-Max\n")
print("Running time for this classifer:",time5,"s")
print("The score of this model is: %f"%score5)
print("\n")
print("\n")

begin_time=time.time()
voting2 = VotingClassifier(estimators=[          #soft
    ('DT_clf',tree.DecisionTreeClassifier()),
    ('SVM_clf',SVC(kernel='rbf', gamma=0.1,probability=True)),
    ('KNN_clf',KNeighborsClassifier(n_neighbors=5)),
    ('MLP_clf',MLPClassifier(max_iter=10))
    ], voting='soft')
voting2=voting2.fit(data,labels)
score6=voting2.score(data_test,labels_test)
end_time=time.time()
time6=end_time-begin_time
print("Ensemble-Mean\n")
print("Running time for this classifer:",time6,"s")
print("The score of this model is: %f"%score6)
print("\n")
print("\n")


