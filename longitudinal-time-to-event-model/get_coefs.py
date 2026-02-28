import pickle, sys
import numpy as np
import pandas as pd


outcome_of_interest = sys.argv[1].strip().lower()

with open(f'data_processed_features_3_{outcome_of_interest}.pickle', 'rb') as f:
    res = pickle.load(f)
names_feat = res['names_feat']
with open(f'results_{outcome_of_interest}.pickle', 'rb') as f:
    res = pickle.load(f)
model_PCAs = res['model_PCAs']
model_Ys = res['model_Ys']
Xmeans = res['X_means']
Xsds = res['X_sds']
Tmeans = res['T_means']
Tsds = res['T_sds']
fold_names = res['CV_names']

"""
PCA --> z-score --> model
y = model(z-score(pca(X)))

PCA:
pc1 = b11 X1 + b12 X2 + ...
pc2 = b21 X1 + b22 X2 + ...

z-score:
t1z = t1/sd_t1
t2z = t2/sd_t2
pc1z = pc1/sd1 + constant1
pc2z = pc2/sd2 + constant2

model:
y ~ c_t1 t1z + c_t2 t2z + c1 pc1z + c2 pc2z + ...
  = c_t1/sd_t1 t1 + c_t2/sd_t2 t2 + c1/sd1 pc1 + c2/sd2 pc2 + ...
  = c_t1/sd_t1 t1 + c_t2/sd_t2 t2 + c1/sd1 (b11 X1 + b12 X2 + ...) + c2/sd2 (b21 X1 + b22 X2 + ...) + ...
  = c_t1/sd_t1 t1 + c_t2/sd_t2 t2 + (c1/sd1*b11+c2/sd2*b21+...) X1 + (c1/sd1*b12+c2/sd2*b22+...) X2 + ...
"""

df_coefs = []
for fi, fn in enumerate(fold_names):
    pca = model_PCAs[fi]
    #t_coef = model_Ys[fi].params[1:(1+3)]/Tsds[fi]
    part1 = model_Ys[fi].params[(1+3):]/Xsds[fi]
    x_coef = (part1.reshape(1,-1) @ pca.components_).flatten()
    #df_coef = pd.DataFrame(data={'FeatureName':names_feat, 'Coef':x_coef})
    df_coefs.append(x_coef)
df_coef = pd.DataFrame(data=np.array(df_coefs).T, index=names_feat, columns=fold_names)
df_coef['Mean'] = df_coef.mean(axis=1)
df_coef['STD'] = df_coef.std(axis=1)
df_coef['CoefVar(STD-Mean Ratio)'] = df_coef.STD/df_coef.Mean
df_coef['SameSign'] = (df_coef.iloc[:,:5]>0).all(axis=1)|(df_coef.iloc[:,:5]<0).all(axis=1)
df_coef = df_coef.sort_values('Mean', key=abs, ascending=False)
print(df_coef)
df_coef.to_excel(f'coefs_{outcome_of_interest}.xlsx')
