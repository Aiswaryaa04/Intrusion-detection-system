# import all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU


df = pd.read_csv('UNSW_NB15_training-set.csv')
df=df.drop(['proto','service','state','attack_cat'], axis=1)
print(df)

X=df.drop(columns=['label'])
y=df['label']
print(X)

names = df.head()
dtree = tree.DecisionTreeClassifier()
rfe = RFE(estimator=dtree, n_features_to_select=16)
rfe.fit(X, y)
# summarize the selection of the attributes
print(rfe.support_)
print("Rank")
print(rfe.ranking_)
print ("Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))
mm=sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))
print(mm)
cols=[mm[0][1],mm[1][1],mm[2][1],mm[3][1],mm[4][1],mm[5][1],mm[6][1],mm[7][1],mm[8][1],mm[9][1],mm[10][1],mm[11][1],mm[12][1],mm[13][1],mm[14][1],mm[15][1]]
print(cols)
print(df[cols])
X=df[cols]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)



sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


 
# Set the n_components=3
principal=PCA(n_components=3)
X_train=principal.fit_transform(X_train)
X_test=principal.fit_transform(X_test)
 
# Check the dimensions of data after PCA
print(X_train)





model= Sequential()

#model.add(Dense(4, kernel_initializer='uniform', activation='relu', input_dim=3))
#model.add(Dense(4, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(3, kernel_initializer='uniform', activation='sigmoid'))
#model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#history=model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
#loss, accuracy = model.evaluate(X_test, y_test)
#print(accuracy)

model.add(Dense(16, kernel_initializer='uniform', activation='relu', input_dim=3))
model.add(Dense(14, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
hist=model.fit(X_train,y_train, epochs=5,batch_size=10,validation_data=(X_test, y_test))

#train and validation loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train','Validation'],loc='upper left')
plt.savefig('results/DNN Loss.png') 
plt.pause(5)
plt.show(block=False)
plt.close()

#train and validation accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train','Validation'],loc='upper left')
plt.savefig('results/DNN Accuracy.png') 
plt.pause(5)
plt.show(block=False)
plt.close()


y_pred=model.predict(X_test)
y_pred = [np.argmax(x) for x in y_pred]


mse=mean_squared_error(y_test, y_pred)
mae=mean_absolute_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
	
	
print("---------------------------------------------------------")
print("MSE VALUE FOR DNN IS %f "  % mse)
print("MAE VALUE FOR DNN IS %f "  % mae)
print("R-SQUARED VALUE FOR DNN IS %f "  % r2)
rms = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE VALUE FOR DNN IS %f "  % rms)
ac=accuracy_score(y_test,y_pred)
print ("ACCURACY VALUE DNN IS %f" % ac)
print("---------------------------------------------------------")

	
result2=open('results/DNNMetrics.csv', 'w')
result2.write("Parameter,Value" + "\n")
result2.write("MSE" + "," +str(mse) + "\n")
result2.write("MAE" + "," +str(mae) + "\n")
result2.write("R-SQUARED" + "," +str(r2) + "\n")
result2.write("RMSE" + "," +str(rms) + "\n")
result2.write("ACCURACY" + "," +str(ac) + "\n")
result2.close()
	
	
df =  pd.read_csv('results/DNNMetrics.csv')
acc = df["Value"]
alc = df["Parameter"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
explode = (0.1, 0, 0, 0, 0)  

fig = plt.figure()
plt.bar(alc, acc,color=colors)
plt.xlabel('Parameter')
plt.ylabel('Value')
plt.title('DNN Metrics Value')
fig.savefig('results/DNNMetricsValue.png') 
plt.pause(5)
plt.show(block=False)
plt.close()

