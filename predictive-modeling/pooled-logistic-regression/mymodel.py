import os, pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.genmod.families.links as sml
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import GroupKFold


def transform_T(T):
    return np.c_[T, T**2, T**3]


def train_(sids, X, T, Y, C, dT, start_params=None, maxiter=100, alpha=None):
    # inner logic
    T2 = transform_T(T)
    XT = np.c_[T2, X]
    XT_mean = XT.mean(axis=0)
    XT_sd = XT.std(axis=0)
    XT_sd[XT_sd==0] = 1
    X_mean = XT_mean[3:]
    X_sd = XT_sd[3:]
    T_mean = XT_mean[:3]
    T_sd = XT_sd[:3]

    # get sIPCW (stablized inverse propensity of censoring weights)
    XT_ = np.c_[np.ones(len(XT)), (XT-XT_mean)/XT_sd]
    model_C1 = sm.GLM( C, XT_, family=sm.families.Binomial(link=sml.CLogLog()), offset=np.log(dT)).fit()
    g = model_C1.predict(XT_, offset=np.log(dT))
    log1mg_den = np.log1p(-np.clip(g, 1e-10, 1-1e-10))

    T_ = np.c_[np.ones(len(T2)), (T2-T_mean)/T_sd]
    model_C2 = sm.GLM( C, T_, family=sm.families.Binomial(link=sml.CLogLog()), offset=np.log(dT)).fit()
    g = model_C2.predict(T_, offset=np.log(dT))
    log1mg_nom = np.log1p(-np.clip(g, 1e-10, 1-1e-10))

    sid2ids = pd.DataFrame(data={'sid':sids}).groupby('sid').indices
    unique_sids = sorted(sid2ids.keys())
    sIPCW = np.zeros(len(X))
    for sid in unique_sids:
        mask = sid2ids[sid]
        assert all(np.diff(T[mask])>0)
        sIPCW[mask] = np.exp(np.r_[0,np.cumsum(log1mg_nom[mask][:-1])] - np.r_[0,np.cumsum(log1mg_den[mask][:-1])])
    #print(f'{np.mean(sIPCW) = }')
    #print(f'{np.median(sIPCW) = }')

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
    if alpha is None:
        model_Y = model_Y.fit(start_params=start_params, maxiter=maxiter)# cov_type="cluster", cov_kwds={"groups": sids})
    else:
        model_Y = model_Y.fit_regularized(alpha=alpha,refit=True)  #somehow it doesn't converge: BracketError

    return model_Y, model_C1, X_mean, X_sd, T_mean, T_sd


def train(sites, sids, X, T, Y, C, dT, n_jobs=1, random_state=2026):
    """
    CV + feature selection
    """
    sid2ids = pd.DataFrame(data={'sid':sids}).groupby('sid').indices
    unique_sites = np.unique(sites)

    def foo(j):
        cindexs = []
        for site in unique_sites:
            trids = sites!=site
            teids = sites==site
            model_Y, model_C1, X_mean, X_sd, T_mean, T_sd = train_(sids[trids], X[trids][:,[j]], T[trids], Y[trids], C[trids], dT[trids])
            IPCW = get_IPCW(model_C1, sids[teids], X[teids][:,[j]], T[teids], dT[teids], X_mean, X_sd, T_mean, T_sd)
            riskX, riskT = get_risk(model_Y, X[teids][:,[j]], X_mean, X_sd, T[teids], T_mean, T_sd)
            cindexs.append(uno_c_index(sids[teids], Y[teids], T[teids], riskX, IPCW))
        return np.mean(cindexs)

    with Parallel(n_jobs=n_jobs, verbose=10) as par:
        perfs_per_feature = par(delayed(foo)(j) for j in range(X.shape[1]))
    with open('tmp_perfs_per_feature.pickle','wb') as ff:
        pickle.dump(perfs_per_feature, ff)
    #with open('tmp_perfs_per_feature.pickle','rb') as ff:
    #    perfs_per_feature = pickle.load(ff)
    #print('!!Reading from tmp_perfs_per_feature.pickle')
    perfs_per_feature = np.array(perfs_per_feature)
    feature_order = np.argsort(perfs_per_feature)[::-1]
    feature_order = feature_order[perfs_per_feature[feature_order]>0.6]
    print(f'{X.shape[1]} features reduced to {len(feature_order)} with univariate C-index>0.6')

    nfs = range(1, min(len(feature_order), 20)+1)
    tridss = []
    teidss = []
    unique_sites = sorted(set(sites))
    for site in unique_sites:
        trids = np.where(sites!=site)[0]
        teids = np.where(sites==site)[0]
        tridss.append(trids)
        teidss.append(teids)

    def foo2(trids, teids, feature_mask):
        model_Y, model_C1, X_mean, X_sd, T_mean, T_sd = train_(sids[trids], X[trids][:,feature_mask], T[trids], Y[trids], C[trids], dT[trids])
        IPCW = get_IPCW(model_C1, sids[teids], X[teids][:,feature_mask], T[teids], dT[teids], X_mean, X_sd, T_mean, T_sd)
        riskX, riskT = get_risk(model_Y, X[teids][:,feature_mask], X_mean, X_sd, T[teids], T_mean, T_sd)
        cindex = uno_c_index(sids[teids], Y[teids], T[teids], riskX, IPCW)
        return cindex

    perfs_nf = []
    for nf in nfs:
        feature_mask = feature_order[:nf]

        with Parallel(n_jobs=n_jobs, verbose=10) as par:
            res = par(delayed(foo2)(trids, teids, feature_mask) for trids, teids in zip(tridss, teidss))
        #if len(perfs_nf)==0 or cindex>perfs_nf[-1]:
        perfs_nf.append(np.mean(res))
        #else:
        #    break
        print(nf, perfs_nf)

    breakpoint()
    best_id = np.argmax(perfs_nf)
    best_nf = nfs[best_id]
    feature_mask = feature_order[:best_nf]
    model_Y, model_C1, X_mean, X_sd, T_mean, T_sd = train_(sids, X[:,feature_mask], T, Y, C, dT)
    return model_Y, model_C1, X_mean, X_sd, T_mean, T_sd, feature_mask, perfs_per_feature, perfs_nf


def predict_survival_curve(model_Y, sids, X, T, dT, X_mean, X_sd, T_mean, T_sd):
    unique_sids = sorted(set(sids))
    T2 = transform_T(T)
    XT = np.c_[T2, X]
    XT_mean = np.r_[T_mean, X_mean]
    XT_sd = np.r_[T_sd, X_sd]
    XT_ = np.c_[np.ones(len(XT)), (XT-XT_mean)/XT_sd]

    g = model_Y.predict(XT_, offset=np.log(dT))
    log1mg = np.log1p(-np.clip(g, 1e-10, 1-1e-10))
    S = np.zeros(len(X))
    for sid in unique_sids:
        mask = sids==sid
        S[mask] = np.exp(np.r_[0,np.cumsum(log1mg[mask][:-1])])
    return S


def get_IPCW(model_C, sids, X, T, dT, X_mean, X_sd, T_mean, T_sd, clip=10):
    unique_sids = sorted(set(sids))
    T2 = transform_T(T)
    XT = np.c_[T2, X]
    XT_mean = np.r_[T_mean, X_mean]
    XT_sd = np.r_[T_sd, X_sd]
    XT_ = np.c_[np.ones(len(XT)), (XT-XT_mean)/XT_sd]

    g = model_C.predict(XT_, offset=np.log(dT))
    log1mg = np.log1p(-np.clip(g, 1e-10, 1-1e-10))
    IPCW = np.zeros(len(X))
    sid2ids = pd.DataFrame(data={'sid':sids}).groupby('sid').indices
    for sid in unique_sids:
        mask = sid2ids[sid]
        IPCW[mask] = np.exp(-np.r_[0,np.cumsum(log1mg[mask][:-1])])
    IPCW = np.clip(IPCW, 1, clip)
    return IPCW


def get_risk(model_Y, X, X_mean, X_sd, T, T_mean, T_sd):
    riskX = (model_Y.params[4:]*(X-X_mean)/X_sd).sum(axis=1)  # 4 for 3+1 (dimension of t + intercept)
    Tt = transform_T(T)
    riskT = (model_Y.params[1:4]*(Tt-T_mean)/T_sd).sum(axis=1)
    return riskX, riskT


def uno_c_index(sids, Y, T, risk, IPCW):
    sid2ids = pd.DataFrame(data={'sid':sids}).groupby('sid').indices
    unique_sids = sorted(sid2ids.keys())
    cindex_den = []
    cindex_nom = []
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
                concord.append(float(risk[ids_i[-1]]>risk[ids_j[-1]]))
        cindex_nom.append(IPCW[ids_i[-1]]**2*sum(concord)/10.)
        cindex_den.append(IPCW[ids_i[-1]]**2*len(concord)/10.)
    cindex = sum(cindex_nom) / sum(cindex_den)
    return cindex


def evaluate(model_Y, model_C, sids_, Y_, X_, T_, X_mean, X_sd, T_mean, T_sd, dT_, nbt=1000, random_state=2026, verbose=True, n_jobs=1):
    perf = {}
    IPCW_ = get_IPCW(model_C, sids_, X_, T_, dT_, X_mean, X_sd, T_mean, T_sd)

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

        # Uno's C-index
        riskX, riskT = get_risk(model_Y, X, X_mean, X_sd, T, T_mean, T_sd)
        cindex = uno_c_index(sids, Y, T, riskX, IPCW)
        return cindex, riskX, riskT

    with Parallel(n_jobs=n_jobs, verbose=False) as par:
        res = par(delayed(_bt)(bti, random_state+bti) for bti in tqdm(range(nbt+1), disable=not verbose))
    perf['CIndex'] = [x[0] for x in res]
    riskX = res[0][1]
    riskT = res[0][2]

    df_perf = pd.DataFrame(columns=['Metric', 'Value', 'LB', 'UB'],#TODO PVal
        data=[ [m, v[0]]+list(np.nanpercentile(v[1:], (2.5,97.5))) for m,v in perf.items() ])
    return df_perf, riskX, riskT
