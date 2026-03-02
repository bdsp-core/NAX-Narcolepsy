import os, pickle, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#from sklearn.pipeline import Pipeline
#from sklearn.frozen import FrozenEstimator
#from sklearn.calibration import CalibratedClassifierCV
#from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from mymodel import train, predict_survival_curve, evaluate


def main():
    outcome_of_interest = sys.argv[1].strip().lower()
    assert outcome_of_interest in ['nt1', 'nt2ih']

    cv_method = 'loso'
    site_names = ['stan', 'bch', 'bidmc', 'emory', 'mgb']

    """
    import pyarrow.parquet as pq
    with pq.ParquetFile('features_first_time/features_3.parquet') as dfold:
        cols = dfold.schema.names
    cols[cols.index('element')]='filename'

    df0 = pd.read_parquet('features_update/control_update/features/features_3.parquet')
    df1 = pd.read_parquet(f'features_update/{outcome_of_interest}/features_3.parquet')
    df1 = df1.rename(columns={'cohort':'site'})
    if outcome_of_interest=='nt1':
        # df1 contains df0, remove
        to_remove_sids = df0.bdsp_patient_id.unique()
        df1 = df1[~np.in1d(df1.bdsp_patient_id,to_remove_sids)].reset_index(drop=True)

    # get site for df0
    df_ = pd.read_parquet('features_update/control_update/controls.parquet')
    df_ = df_.drop_duplicates(ignore_index=True)
    df_['site'] = df_.source.str.split('_',expand=True)[0]
    df0 = df0.merge(df_,on='bdsp_patient_id',how='left')
    assert df0.site.notna().all()
    df = pd.concat([df0[cols], df1[cols]], axis=0, ignore_index=True)


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
    with open(f'data_processed_features_3_{outcome_of_interest}.pickle', 'wb') as f:
        pickle.dump({'sites':sites, 'sids':sids, 'Y':Y, 'T':T, 'X':X, 'C':C, 'dT':dT, 'names_feat':names_feat}, f)
    """
    with open(f'data_processed_features_3_{outcome_of_interest}.pickle', 'rb') as f:
        res = pickle.load(f)
    sites = res['sites']
    sids = res['sids']
    X = res['X']
    T = res['T']
    Y = res['Y']
    C = res['C']
    dT = res['dT']
    names_feat = res['names_feat']

    good_feat_ids = np.where((np.abs(X)>0).sum(axis=0)>=50)[0]
    print(f'{X.shape[1]} features reduced to {len(good_feat_ids)} due to having <50 non-zeros.')
    X = X[:,good_feat_ids]
    names_feat = [names_feat[x] for x in good_feat_ids]

    sid2ids = pd.Series(sids).to_frame().groupby(0).indices
    unique_sids = sorted(sid2ids.keys())
    unique_sites = pd.Series([sites[sid2ids[sid][-1]] for sid in unique_sids])
    unique_Ys = pd.Series([Y[sid2ids[sid][-1]] for sid in unique_sids])
    for site in site_names:
        print(f'{site}: n={(unique_sites==site).sum()} %outcome={(unique_Ys[unique_sites==site]==1).mean()*100:.2f}%')

    cv_folds_var = sites
    unique_cv_folds = site_names

    n_jobs = 14
    #pcas = []
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
    feature_masks_cv = []
    perf_per_feats = []
    perf_nfs = []
    for cvi, cvf in enumerate(unique_cv_folds):
        print(f'\n========\nCV fold {cvi+1}/{len(unique_cv_folds)} {cvf}\n========\n')
        ids_tr = cv_folds_var!=cvf
        sitestr = sites[ids_tr]
        sidstr = sids[ids_tr]

        Xtr = X[ids_tr]
        Ttr = T[ids_tr]
        Ytr = Y[ids_tr]
        Ctr = C[ids_tr]
        dTtr = dT[ids_tr]
        #pca = PCA(n_components=60, random_state=2026+cvi).fit(Xtr) # 60 is roughly 95%
        #print(f'{pca.explained_variance_ratio_.sum() = }')
        #Xtr2 = pca.transform(Xtr)
        model_Y, model_C, X_mean, X_sd, T_mean, T_sd, selected_feature_mask, perf_per_feat, perf_nf = train(sitestr, sidstr, Xtr, Ttr, Ytr, Ctr, dTtr, n_jobs=n_jobs, random_state=2026+cvi)
        #X_mean2 = X_mean[selected_feature_mask]
        #X_sd2 = X_sd[selected_feature_mask]

        ids_te = ~ids_tr
        sidste = sids[ids_te]
        Xte = X[ids_te]
        Tte = T[ids_te]
        Yte = Y[ids_te]
        Cte = C[ids_te]
        dTte = dT[ids_te]
        #Xte2 = pca.transform(Xte)
        Xte2 = Xte[:,selected_feature_mask]
        Spte = predict_survival_curve(model_Y, sidste, Xte2, Tte, dTte, X_mean, X_sd, T_mean, T_sd)
        df_perf_te, riskX_te, riskT_te = evaluate(model_Y, model_C, sidste, Yte, Xte2, Tte, X_mean, X_sd, T_mean, T_sd, dTte, n_jobs=n_jobs)
        df_perf_te.insert(0, 'FoldHoldOut', cvf)
        perf_te_folds.append(df_perf_te)
        print(df_perf_te)
        Sp_cv[ids_te] = Spte
        riskXs_cv[ids_te] = riskX_te
        riskTs_cv[ids_te] = riskT_te

        #pcas.append(pca)
        model_Cs.append(model_C)
        #model_C_noms.append(model_C_nom)
        model_Ys.append(model_Y)
        X_means.append(X_mean)
        X_sds.append(X_sd)
        T_means.append(T_mean)
        T_sds.append(T_sd)
        feature_masks_cv.append(selected_feature_mask)
        perf_per_feats.append(perf_per_feat)
        perf_nfs.append(perf_nf)
    
    with open(f'results_{outcome_of_interest}_CV{cv_method}.pickle', 'wb') as f:
        pickle.dump({#'model_PCAs':pcas, 
            'feature_masks_cv':feature_masks_cv, 'perf_per_feats':perf_per_feats, 'perf_nfs':perf_nfs,
            'model_Cs':model_Cs, 'model_Ys':model_Ys,
            'X_names':names_feat, 'X_means':X_means, 'X_sds':X_sds, 'T_means':T_means, 'T_sds':T_sds,
            'CV_names':unique_cv_folds, 'perf_te_folds':perf_te_folds,
            'S_pred_cv':Sp_cv, 'riskX_cv':riskXs_cv, 'riskT_cv':riskTs_cv}, f)

    #perf_tr_cv = pd.concat(perf_tr_folds, axis=0, ignore_index=True)
    perf_te_cv = pd.concat(perf_te_folds, axis=0, ignore_index=True)
    for x in ['bch','bidmc','emory','mgb','stan']:perf_te_cv.loc[perf_te_cv.FoldHoldOut==x,'N_subject']=len(set(sids[sites==x]))
    for x in ['bch','bidmc','emory','mgb','stan']:perf_te_cv.loc[perf_te_cv.FoldHoldOut==x,'N_row']=len(sids[sites==x])
    #perf_tr_cv.to_csv(f'perf_tr_features_3_{outcome_of_interest}.csv', index=False)
    print(perf_te_cv)
    perf_te_cv.to_csv(f'perf_cv_features_3_{outcome_of_interest}_CV{cv_method}.csv', index=False)


if __name__=='__main__':
    main()

