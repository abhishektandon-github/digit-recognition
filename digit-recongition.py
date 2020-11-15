from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from sklearn.model_selection import train_test_split

dataset=load_digits()
print(dataset.keys())

sb.heatmap(dataset.images[90],annot=True,cmap='gray')
plt.show()

print(dataset.target_names)
#print(dataset.data[0])

#sb.countplot(x=dataset.target)
#plt.xticks(dataset.target,dataset.target_names)
#plt.show()

X_train,X_test,y_train,y_test=train_test_split(dataset.data,dataset.target,test_size=0.2,random_state=10)

print(X_train.shape)
print(X_test.shape)

model = LogisticRegression()
model.fit(X_train,y_train)


y_pred=model.predict(X_test)
#cm = confusion_matrix(y_test,y_pred)
#sb.heatmap(data=cm,annot=True,xticklabels=dataset.target_names,yticklabels=dataset.target_names)
#plt.show()

print(accuracy_score(y_test,y_pred))


data=model.predict([dataset.data[90]])
print(data)











