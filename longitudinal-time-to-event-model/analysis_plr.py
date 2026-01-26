import os, pickle
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#from sklearn.pipeline import Pipeline
#from sklearn.frozen import FrozenEstimator
#from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
import statsmodels.api as sm
import statsmodels.genmod.families.links as sml



def transform_T(T):
    return np.c_[T, T**2, T**3]


def train(sites, sids, X, T, Y, C, dT):
    T2 = transform_T(T)
    XT = np.c_[T2, X]
    XT_mean = XT.mean(axis=0)
    XT_sd = XT.std(axis=0)
    X_mean = XT_mean[3:]
    X_sd = XT_sd[3:]
    T_mean = XT_mean[:3]
    T_sd = XT_sd[:3]
    unique_sids = sorted(set(sids))

    # get sIPCW (stablized inverse propensity of censoring weights)
    XT_ = sm.add_constant((XT-XT_mean)/XT_sd)
    model_C1 = sm.GLM( C, XT_, family=sm.families.Binomial(link=sml.CLogLog()), offset=np.log(dT)).fit()
    g = model_C1.predict(XT_, offset=np.log(dT))
    log1mg_den = np.log1p(-np.clip(g, 1e-10, 1-1e-10))

    T_ = sm.add_constant((T2-T_mean)/T_sd)
    model_C2 = sm.GLM( C, T_, family=sm.families.Binomial(link=sml.CLogLog()), offset=np.log(dT)).fit()
    g = model_C2.predict(T_, offset=np.log(dT))
    log1mg_nom = np.log1p(-np.clip(g, 1e-10, 1-1e-10))

    sIPCW = np.zeros(len(X))
    for sid in unique_sids:
        mask = sids==sid
        assert all(np.diff(T[mask])>0)
        sIPCW[mask] = np.exp(np.r_[0,np.cumsum(log1mg_nom[mask][:-1])] - np.r_[0,np.cumsum(log1mg_den[mask][:-1])])
    print(f'{sIPCW.mean() = }')

    #sw_sid = np.zeros(len(Xtr))
    #usid = np.unique(sidstr)
    #for x in usid:
    #    sw_sid[sidstr==x] = 1/np.sum(sidstr==x)
    #sw_y = np.zeros(len(Xtr))
    #sw_y[Ytr==0] = 1/np.sum(Ytr==0)
    #sw_y[Ytr==1] = 1/np.sum(Ytr==1)
    #sw = sw_sid# * sw_y

    sw = sIPCW
    sw = np.clip(sw,0.02,50)
    sw = sw/sw.mean()

    #model = LogisticRegression(C=1, l1_ratio=0.5, solver='saga')
    #model.fit(XT, Y, sample_weight=sw)
    #model = CalibratedClassifierCV(estimator=FrozenEstimator(model), method='sigmoid')
    #model.fit(XT, Y)
    model_Y = sm.GLM(Y, XT_, family=sm.families.Binomial(link=sml.CLogLog()), offset=np.log(dT), var_weights=sw)
    model_Y = model_Y.fit( cov_type="cluster", cov_kwds={"groups": sids})

    return model_Y, model_C1, X_mean, X_sd, T_mean, T_sd


def predict_survival_curve(model_Y, sids, X, T, dT, X_mean, X_sd, T_mean, T_sd):
    unique_sids = sorted(set(sids))
    T2 = transform_T(T)
    XT = np.c_[T2, X]
    XT_mean = np.r_[T_mean, X_mean]
    XT_sd = np.r_[T_sd, X_sd]
    XT_ = sm.add_constant((XT-XT_mean)/XT_sd)

    g = model_Y.predict(XT_, offset=np.log(dT))
    log1mg = np.log1p(-np.clip(g, 1e-10, 1-1e-10))
    S = np.zeros(len(X))
    for sid in unique_sids:
        mask = sids==sid
        S[mask] = np.exp(np.r_[0,np.cumsum(log1mg[mask][:-1])])
    return S


def get_IPCW(model_C, sids, X, T, dT, X_mean, X_sd, T_mean, T_sd, clip=50):
    unique_sids = sorted(set(sids))
    T2 = transform_T(T)
    XT = np.c_[T2, X]
    XT_mean = np.r_[T_mean, X_mean]
    XT_sd = np.r_[T_sd, X_sd]
    XT_ = sm.add_constant((XT-XT_mean)/XT_sd)

    g = model_C.predict(XT_, offset=np.log(dT))
    log1mg = np.log1p(-np.clip(g, 1e-10, 1-1e-10))
    IPCW = np.zeros(len(X))
    for sid in unique_sids:
        mask = sids==sid
        IPCW[mask] = np.exp(-np.r_[0,np.cumsum(log1mg[mask][:-1])])
    IPCW = np.clip(IPCW, 1, clip)
    return IPCW


def evaluate(model_Y, sids_, Y_, X_, T_, X_mean, X_sd, T_mean, T_sd, IPCW_, nbt=1000, random_state=2026, verbose=True, n_jobs=1):
    perf = {}

    def _bt(bti, random_state):
        if bti==0:
            btids = np.arange(len(sids_))
            sids = sids_
        else:
            # bootstrap subjects
            np.random.seed(random_state)
            unique_sids = sorted(set(sids_))
            unique_sids_bt = np.random.choice(unique_sids, len(unique_sids), replace=True)
            btids = []; sids = []
            for ii, sid in enumerate(unique_sids_bt):
                x = np.where(sids_==sid)[0]
                sids.extend([ii]*len(x))
                btids.extend(x)
        Y = Y_[btids]
        X = X_[btids]
        T = T_[btids]
        IPCW = IPCW_[btids]
        sid2ids = pd.DataFrame(data={'sid':sids}).groupby('sid').indices
        unique_sids = sorted(sid2ids.keys())

        # Uno's C-index
        cindex_den = 0.
        cindex_nom = 0.
        riskX = (model_Y.params[4:]*(X-X_mean)/X_sd).sum(axis=1)  # 4 for 3+1 (dimension of t + intercept)
        Tt = transform_T(T)
        riskT = (model_Y.params[1:4]*(Tt-T_mean)/T_sd).sum(axis=1)
        for ii in range(len(unique_sids)):
            ids_i = sid2ids[unique_sids[ii]]
            if Y[ids_i[-1]]==0: continue
            concord = []
            for jj in range(len(unique_sids)):
                if ii==jj:continue
                ids_j = sid2ids[unique_sids[jj]]
                if T[ids_j[-1]] < T[ids_i[-1]]:continue # limit to all i's with shorter time-to-event/censoring, more severe, higher risk
                if T[ids_j[-1]] == T[ids_i[-1]]:
                    concord.append(0.5)
                else:
                    concord.append(float(riskX[ids_i[-1]]>riskX[ids_j[-1]]))
            cindex_nom += IPCW[ids_i[-1]]**2*sum(concord)/10.
            cindex_den += IPCW[ids_i[-1]]**2*len(concord)/10.
        cindex = cindex_nom / cindex_den
        return cindex, riskX, riskT

    with Parallel(n_jobs=n_jobs, verbose=False) as par:
        res = par(delayed(_bt)(bti, random_state+bti) for bti in tqdm(range(nbt+1), disable=not verbose))
    perf['CIndex'] = [x[0] for x in res]
    riskX = res[0][1]
    riskT = res[0][2]

    df_perf = pd.DataFrame(columns=['Metric', 'Value', 'LB', 'UB'],#TODO PVal
        data=[ [m, v[0]]+list(np.nanpercentile(v[1:], (2.5,97.5))) for m,v in perf.items() ])
    return df_perf, riskX, riskT


def main():
    """
    df = pd.read_parquet('features_3.parquet')
    sitename = 'site'
    sidname = 'bdsp_patient_id'
    yname = 'n+_state'
    tname = 'days_since_first_visit'
    names_feat = list(df.columns)[6:]

    # ensure the last row in one subject has Y=0 or Y=the first 1
    sid2ids = df.groupby(sidname).indices
    exclude_ids = []
    for sid, ids in tqdm(sid2ids.items(), total=len(sid2ids)):
        assert df.loc[ids[0],tname]==0
        if len(ids)==1:
            exclude_ids.extend(ids)
            continue
        assert all(df.loc[ids,tname].diff().iloc[1:]>0)
        assert df.loc[ids,sitename].nunique()==1
        yy = df.loc[ids, yname].values
        if yy.max()==1:
            cutoff_point = np.where(yy==1)[0][0]+1
            if cutoff_point==1:
                exclude_ids.extend(ids) # exclude all records if only 1 row after ensuring last row is 0/1
            else:
                exclude_ids.extend( ids[cutoff_point:] )
        else:
            cutoff_point = len(ids)
        if cutoff_point>1:
            tt = df.loc[ids[:cutoff_point], tname].values  # exclude all records if any gap >=5 years
            if any(np.diff(tt)/365.25>=5):
                exclude_ids.extend(ids)
    keep_mask = np.ones(len(df), dtype=bool)
    keep_mask[sorted(set(exclude_ids))] = False
    print(f'Before ensuring last row, {df.shape = }, N(subject) = {df[sidname].nunique()}')
    df = df[keep_mask].reset_index(drop=True)
    print(f'After ensuring last row, {df.shape = }, N(subject) = {df[sidname].nunique()}')

    # discretize to every 2 weeks (14 days)
    #DT = 14
    #     *2 weeks* based results from for sid,ids in sid2ids.items():dts.extend(np.diff(df[tname][ids]))
    #     pd.DataFrame(data={'%':[0,1,5,10,25,50,75,90,95,99,100], 'dt':np.percentile(dts,(0,1,5,10,25,50,75,90,95,99,100))*52})
    #     %           dt
    #    25     0.147844
    #    50     0.739220  --> roughly 1, but to save space, use 2 weeks
    #    75     2.365503
    #sid2ids = df.groupby(sidname).indices
    #df2 = []
    #for sid,ids in sid2ids.items():
    #    df_ = df.iloc[ids].reset_index(drop=True)
    #    #if len(ids)==1: df2.append(df_)
    #    #else:
    #    #29 --> 29.01/14 = 2.07 --> 3 --> [0,14), [14,28), [28,42)
    #    #28 --> 28.01/14 = 2.01 --> 3 --> [0,14), [14,28), [28,42)
    #    #27 --> 27.01/14 = 1.93 --> 2 --> [0,14), [14,28)
    #    max_ = int(np.ceil((float(df_[tname].max())+0.01)/DT))
    #    # linear interpolation to get features at fixed intervals
    #    df_res = df_.iloc[[0]*max_].reset_index(drop=True).copy()
    #    ts = np.arange(max_)
    #    for col in names_feat:
    #       df_res[col] = np.interp(ts*DT, df_[tname].values, df_[col].values)
    #    df_res[yname] = [0 if t<max_-1 else df_[yname].iloc[-1] for t in ts]
    #    df2.append(df_res)
    #df = pd.concat(df2, axis=0, ignore_index=True).copy()
    #print(f'After discretizing time to weeks, N(row) = {len(df)}, N(subject) = {df[sidname].nunique()}')

    # create censoring variable
    sid2ids = df.groupby(sidname).indices
    C = np.zeros(len(df), dtype=int)
    dT = np.zeros(len(df))+np.nan
    for sid, ids in sid2ids.items():
        C[ids[-1]] = int(df.loc[ids[-1], yname]==0)
        dT[ids[1:]] = np.diff(df.loc[ids, tname].values)/365.25

    # get variables
    sites = df[sitename].values
    sids = df[sidname].values
    Y = df[yname].values
    T = df[tname].values/365.25
    X = df.iloc[:,6:].values
    #assert np.isnan(X).sum()==0

    ids = ~np.isnan(dT)
    sites = sites[ids]
    sids = sids[ids]
    X = X[ids]
    T = T[ids]
    Y = Y[ids]
    C = C[ids]
    dT = dT[ids]

    assert len(names_feat)==X.shape[1]
    with open('data_processed_features_3.pickle', 'wb') as f:
        pickle.dump({'sites':sites, 'sids':sids, 'Y':Y, 'T':T, 'X':X, 'C':C, 'dT':dT, 'names_feat':names_feat}, f)
    """
    with open('data_processed_features_3.pickle', 'rb') as f:
        res = pickle.load(f)
    sites = res['sites']
    sids = res['sids']
    X = res['X']
    T = res['T']
    Y = res['Y']
    C = res['C']
    dT = res['dT']
    names_feat = res['names_feat']
    """
    pd.DataFrame(data={'site':sites,'sid':sids}).drop_duplicates('sid').site.value_counts()
stan     695
bidmc    664
bch      637
emory    538
mgb      291
    """
    n_jobs = 14

    unique_cv_folds = sorted(set(sites))
    pcas = []
    model_Cs = []
    #model_C_noms = []
    model_Ys = []
    X_means = []
    X_sds = []
    T_means = []
    T_sds = []
    perf_tr_folds = []
    perf_te_folds = []
    Sp_cv = np.zeros(len(X))
    riskXs_cv = np.zeros(len(X))
    riskTs_cv = np.zeros(len(T))
    for cvi, cvf in enumerate(unique_cv_folds):
        print(f'\n========\nCV fold {cvi+1}/{len(unique_cv_folds)} {cvf}\n========\n')
        ids_tr = sites!=cvf
        sitestr = sites[ids_tr]
        sidstr = sids[ids_tr]

        Xtr = X[ids_tr]
        Ttr = T[ids_tr]
        Ytr = Y[ids_tr]
        Ctr = C[ids_tr]
        dTtr = dT[ids_tr]
        pca = PCA(n_components=60, random_state=2026+cvi).fit(Xtr) # 60 is roughly 95%
        print(f'{pca.explained_variance_ratio_.sum() = }')
        Xtr2 = pca.transform(Xtr)
        model_Y, model_C, X_mean, X_sd, T_mean, T_sd = train(sitestr, sidstr, Xtr2, Ttr, Ytr, Ctr, dTtr)
        IPCWtr = get_IPCW(model_C, sidstr, Xtr2, Ttr, dTtr, X_mean, X_sd, T_mean, T_sd)
        Sptr = predict_survival_curve(model_Y, sidstr, Xtr2, Ttr, dTtr, X_mean, X_sd, T_mean, T_sd)
        #df_perf_tr, riskX_tr, riskT_tr = evaluate(model_Y, sidstr, Ytr, Xtr2, Ttr, X_mean, X_sd, T_mean, T_sd, IPCWtr, n_jobs=n_jobs)
        #df_perf_tr.insert(0, 'FoldTrain', cvf)
        #perf_tr_folds.append(df_perf_tr)
        #print(df_perf_tr)
       
        ids_te = ~ids_tr
        sidste = sids[ids_te]
        Xte = X[ids_te]
        Tte = T[ids_te]
        Yte = Y[ids_te]
        Cte = C[ids_te]
        dTte = dT[ids_te]
        Xte2 = pca.transform(Xte)
        Spte = predict_survival_curve(model_Y, sidste, Xte2, Tte, dTte, X_mean, X_sd, T_mean, T_sd)
        IPCWte = get_IPCW(model_C, sidste, Xte2, Tte, dTte, X_mean, X_sd, T_mean, T_sd)
        df_perf_te, riskX_te, riskT_te = evaluate(model_Y, sidste, Yte, Xte2, Tte, X_mean, X_sd, T_mean, T_sd, IPCWte, n_jobs=n_jobs)
        df_perf_te.insert(0, 'FoldHoldOut', cvf)
        perf_te_folds.append(df_perf_te)
        print(df_perf_te)
        Sp_cv[ids_te] = Spte
        riskXs_cv[ids_te] = riskX_te
        riskTs_cv[ids_te] = riskT_te

        pcas.append(pca)
        model_Cs.append(model_C)
        #model_C_noms.append(model_C_nom)
        model_Ys.append(model_Y)
        X_means.append(X_mean)
        X_sds.append(X_sd)
        T_means.append(T_mean)
        T_sds.append(T_sd)

    with open('results.pickle', 'wb') as f:
        pickle.dump({'model_PCAs':pcas, 'model_Cs':model_Cs, 'model_Ys':model_Ys,
            'X_means':X_means, 'X_sds':X_sds, 'T_means':T_means, 'T_sds':T_sds,
            'CV_names':unique_cv_folds,
            'S_pred_cv':Sp_cv, 'riskX_cv':riskXs_cv, 'riskT_cv':riskTs_cv}, f)

    #perf_tr_cv = pd.concat(perf_tr_folds, axis=0, ignore_index=True)
    perf_te_cv = pd.concat(perf_te_folds, axis=0, ignore_index=True)
    for x in ['bch','bidmc','emory','mgb','stan']:perf_te_cv.loc[perf_te_cv.FoldHoldOut==x,'N_subject']=len(set(sids[sites==x]))
    for x in ['bch','bidmc','emory','mgb','stan']:perf_te_cv.loc[perf_te_cv.FoldHoldOut==x,'N_row']=len(sids[sites==x])
    #perf_tr_cv.to_csv('perf_tr_features_3.csv', index=False)
    perf_te_cv.to_csv('perf_cv_features_3.csv', index=False)


if __name__=='__main__':
    main()

