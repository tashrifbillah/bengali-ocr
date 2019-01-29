import pandas as pd



ref= pd.read_csv("sampleSubmission.csv")
file_ref = ref['key'].values
# label_ref= ref['label'].values

res=  pd.read_csv("prediction_" +str(index)+ ".csv")
file_res = res['key'].values
label_res= res['label'].values


label_ref= np.array([label_res[file_res==f] for f in file_ref])
ref['label']= label_ref.flatten( )
ref.to_csv("sequenced_prediction_"+ str(index)+".csv", index=False)



