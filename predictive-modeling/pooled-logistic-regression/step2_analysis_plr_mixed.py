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

    cv_method = 'mixed'
    site_names = ['stan', 'bch', 'bidmc', 'emory', 'mgb']

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

    #cv_folds_var = sites
    #unique_cv_folds = site_names
    sids_tr = []
    for site in site_names:
        unique_sids_this_site = np.unique(sids[sites==site])
        unique_Ys_this_site = [ Y[sid2ids[sid][-1]] for sid in unique_sids_this_site]
        skf = StratifiedKFold(n_splits=2, random_state=2026, shuffle=True)
        for _, (trids_, _) in enumerate(skf.split(unique_sids_this_site.reshape(-1,1), unique_Ys_this_site)):
            sids_tr.extend(unique_sids_this_site[trids_])
            break
    ids_tr = np.in1d(sids, sids_tr)

    n_jobs = 14
    sitestr = sites[ids_tr]
    sidstr = sids[ids_tr]
    Xtr = X[ids_tr]
    Ttr = T[ids_tr]
    Ytr = Y[ids_tr]
    Ctr = C[ids_tr]
    dTtr = dT[ids_tr]
    model_Y, model_C, X_mean, X_sd, T_mean, T_sd, selected_feature_mask, perf_per_feat, perf_nf = train(sitestr, sidstr, Xtr, Ttr, Ytr, Ctr, dTtr, n_jobs=n_jobs, random_state=2026)

    ids_te = ~ids_tr
    sidste = sids[ids_te]
    Xte = X[ids_te]
    Tte = T[ids_te]
    Yte = Y[ids_te]
    Cte = C[ids_te]
    dTte = dT[ids_te]
    Xte2 = Xte[:,selected_feature_mask]
    Spte = predict_survival_curve(model_Y, sidste, Xte2, Tte, dTte, X_mean, X_sd, T_mean, T_sd)
    df_perf_te, riskX_te, riskT_te = evaluate(model_Y, model_C, sidste, Yte, Xte2, Tte, X_mean, X_sd, T_mean, T_sd, dTte, n_jobs=n_jobs)
    print(df_perf_te)
    breakpoint()
    print('####risk super big?')

    with open(f'results_{outcome_of_interest}_CV{cv_method}.pickle', 'wb') as f:
        pickle.dump({#'model_PCAs':pcas, 
            'ids_tr':np.where(ids_tr)[0], 'ids_te':np.where(ids_te)[0],
            'CV_names':['train'],
            'feature_masks_cv':[selected_feature_mask], 'perf_per_feats':[perf_per_feat], 'perf_nfs':[perf_nf],
            'model_Cs':[model_C], 'model_Ys':[model_Y],
            'X_names':names_feat, 'X_means':[X_mean], 'X_sds':[X_sd], 'T_means':[T_mean], 'T_sds':[T_sd],
            'perf_te_folds':[df_perf_te],
            'S_pred_cv':[Spte], 'riskX_cv':[riskX_te], 'riskT_cv':[riskT_te]}, f)

    df_perf_te.to_csv(f'perf_cv_features_3_{outcome_of_interest}_CV{cv_method}.csv', index=False)


if __name__=='__main__':
    main()

