import pickle, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import seaborn as sns
sns.set_style('ticks')


def get_pointestimate_ci(X, nbt=1000, random_state=2026, verbose=False):
    np.random.seed(random_state)
    vals = []
    for bti in tqdm(range(nbt+1), disable=not verbose):
        if bti==0:
            Xbt = X
        else:
            btids = np.random.choice(len(X), len(X), replace=True)
            Xbt = X[btids]
        vals.append(np.nanmean(Xbt, axis=0))
    val = vals[0]
    lb, ub = np.nanpercentile(np.array(vals[1:]), (2.5,97.5), axis=0)

    return val, lb, ub


def main():
    outcome_of_interest = sys.argv[1].lower().strip()
    cv_method = sys.argv[2].strip().lower()

    with open(f'data_processed_features_3_{outcome_of_interest}.pickle', 'rb') as f:
        res = pickle.load(f)
    sites = res['sites']
    sids = res['sids']
    X = res['X']
    T = res['T']
    Y = res['Y']
    C = res['C']
    dT = res['dT']
    with open(f'results_{outcome_of_interest}_CV{cv_method}.pickle', 'rb') as f:
        res = pickle.load(f)
    riskX = res['riskX_cv']
    riskT = res['riskT_cv']
    model_Ys = res['model_Ys']
    T_means = res['T_means']
    T_sds = res['T_sds']

    sid2ids = pd.DataFrame(data={'sid':sids}).groupby('sid').indices
    unique_sids = sorted(sid2ids.keys())

    Tstart = -4
    Tend = 0
    Tstep = 0.1
    ts = np.arange(Tstart, Tend+Tstep, Tstep)
    N_ts = len(ts)
    riskXs = []
    riskTs = []
    base_riskXs = []
    last_Ys = []
    #np.random.seed(2026)
    for sid in unique_sids:
        site = sorted(set(sites[sids==sid]))
        assert len(site)==1
        site = site[0]
        siteid = res['CV_names'].index(site)
        ids = sid2ids[sid]
        T_ = T[ids]
        #if Y[ids[-1]]==1:
        tt = T_[-1] + ts
        #else:
        #    tt = T_[-1] + np.random.rand(N_ts)*(Tend - Tstart) + Tstart

        riskX_ = riskX[ids]
        fooX = interp1d(T_, riskX_, bounds_error=False)
        riskXs.append(fooX(tt))

        #riskT_ = riskT[ids]
        #fooT = interp1d(T_, riskT_, kind='cubic', bounds_error=False)
        #riskTs.append(fooT(tt))
        riskTs.append((model_Ys[siteid].params[1:4]*(np.c_[tt,tt**2,tt**3]-T_means[siteid])/T_sds[siteid]).sum(axis=1))

        base_riskXs.append(riskX_[0])
        last_Ys.append(Y[ids[-1]])
    riskXs = np.array(riskXs)
    riskTs = np.array(riskTs)
    risks = riskXs + riskTs
    base_riskXs = np.array(base_riskXs)
    last_Ys = np.array(last_Ys)

    #lb, ub = np.nanpercentile(base_riskXs, (100/3,200/3))
    #mask_high = base_riskXs>ub
    #mask_low = base_riskXs<lb
    mask_high = last_Ys==1
    mask_low = last_Ys==0

    """
    risk1, risk1_lb, risk1_ub = get_pointestimate_ci(risks[mask_high])
    risk0, risk0_lb, risk0_ub = get_pointestimate_ci(risks[mask_low])
    risk1_ = []
    for ii in tqdm(range(risks.shape[1])):
        xx = risks[mask_high][:,ii]
        xx = xx[~np.isnan(xx)]
        risk1_.append( get_pointestimate_ci(xx))
    risk1 = np.array([x[0] for x in risk1_])
    risk1_lb = np.array([x[1] for x in risk1_])
    risk1_ub = np.array([x[2] for x in risk1_])
    risk0_ = []
    for ii in tqdm(range(risks.shape[1])):
        xx = risks[mask_low][:,ii]
        xx = xx[~np.isnan(xx)]
        risk0_.append( get_pointestimate_ci(xx))
    risk0 = np.array([x[0] for x in risk0_])
    risk0_lb = np.array([x[1] for x in risk0_])
    risk0_ub = np.array([x[2] for x in risk0_])

    risk1 = np.nanmean(risks[mask_high], axis=0)
    risk1_lb = risk1 - np.nanstd(risks[mask_high], axis=0)/np.sqrt((~np.isnan(risks[mask_high])).sum(axis=0))*2
    risk1_ub = risk1 + np.nanstd(risks[mask_high], axis=0)/np.sqrt((~np.isnan(risks[mask_high])).sum(axis=0))*2
    risk0 = np.nanmean(risks[mask_low], axis=0)
    risk0_lb = risk0 - np.nanstd(risks[mask_low], axis=0)/np.sqrt((~np.isnan(risks[mask_low])).sum(axis=0))*2
    risk0_ub = risk0 + np.nanstd(risks[mask_low], axis=0)/np.sqrt((~np.isnan(risks[mask_low])).sum(axis=0))*2
    """
    risk1, risk1_lb25, risk1_ub75, risk1_lb2_5, risk1_ub97_5 = np.nanpercentile(risks[mask_high], (50,25,75,2.5,97.5), axis=0)
    risk0, risk0_lb25, risk0_ub75, risk0_lb2_5, risk0_ub97_5 = np.nanpercentile(risks[mask_low], (50,25,75,2.5,97.5), axis=0)
    Nbt = 1000
    #"""
    auroc = []; auroc_lb = []; auroc_ub = []
    auprc = []; auprc_lb = []; auprc_ub = []; auprc_bl = []
    for ii in tqdm(range(risks.shape[1])):
        np.random.seed(2026+ii)
        df_ = pd.DataFrame(data={
            'Y':np.r_[np.zeros(mask_low.sum()),np.ones(mask_high.sum())],
            'X':np.r_[risks[mask_low][:,ii],risks[mask_high][:,ii]]}).dropna()
        auprc_bl.append(df_.Y.mean())
        aurocs = []
        auprcs = []
        for bti in range(Nbt+1):
            if bti==0:
                Xbt = df_.X.values
                Ybt = df_.Y.values
            else:
                btids = np.random.choice(len(df_), len(df_), replace=True)
                Xbt = df_.X.values[btids]
                Ybt = df_.Y.values[btids]
            aurocs.append(roc_auc_score(Ybt, Xbt))
            auprcs.append(average_precision_score(Ybt, Xbt))
        auroc.append(aurocs[0])
        a, b = np.percentile(aurocs[1:], (2.5, 97.5))
        auroc_lb.append(a)
        auroc_ub.append(b)
        auprc.append(auprcs[0])
        a, b = np.percentile(auprcs[1:], (2.5, 97.5))
        auprc_lb.append(a)
        auprc_ub.append(b)
    with open(f'auroc_auprc_Nbt{Nbt}_{outcome_of_interest}_{cv_method}.pickle', 'wb') as f:
        pickle.dump({'auroc':auroc, 'auroc_lb':auroc_lb, 'auroc_ub':auroc_ub,
            'auprc':auprc, 'auprc_lb':auprc_lb, 'auprc_ub':auprc_ub,'auprc_bl':auprc_bl}, f)
    #"""
    #with open(f'auroc_auprc_Nbt{Nbt}_{outcome_of_interest}.pickle', 'rb') as f:
    #    res = pickle.load(f)
    #auroc = res['auroc']; auroc_lb = res['auroc_lb']; auroc_ub = res['auroc_ub']
    #auprc = res['auprc']; auprc_lb = res['auprc_lb']; auprc_ub = res['auprc_ub']; auprc_bl = res['auprc_bl']
    
    plt.close()
    fig = plt.figure(figsize=(8,7.5))
    gs = fig.add_gridspec(4,1,height_ratios=[2,1,1,1])

    ax = fig.add_subplot(gs[0]); ax0 = ax
    #ax.fill_between(ts, risk1_lb2_5, risk1_ub97_5, color='r', alpha=0.3)
    ax.fill_between(ts, risk1_lb25, risk1_ub75, color='r', alpha=0.3)
    ax.plot(ts, risk1, c='r', label='Eventually NT1/2')
    #ax.fill_between(ts, risk0_lb2_5, risk0_ub97_5, color='b', alpha=0.3)
    ax.fill_between(ts, risk0_lb25, risk0_ub75, color='b', alpha=0.3)
    ax.plot(ts, risk0, c='b', label='Never NT1/2')
    ax.legend(loc='lower left', ncols=2)
    ax.set_xlim(Tstart, Tend)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_ylabel('Predicted\nrisk (a.u.)')
    ax.grid(True)
    sns.despine()

    ax = fig.add_subplot(gs[1], sharex=ax0)
    ax.fill_between(ts, auroc_lb, auroc_ub, color='k', alpha=0.3)
    ax.plot(ts, auroc, c='k')
    ax.axhline(0.5, c='k', ls='--')
    ax.set_ylabel('AUROC')
    #ax.set_ylim(0.6,0.83)
    ax.set_yticks([0.6,0.7,0.8])
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.grid(True)
    sns.despine()

    ax = fig.add_subplot(gs[2], sharex=ax0)
    ax.fill_between(ts, auprc_lb, auprc_ub, color='k', alpha=0.3)
    ax.plot(ts, auprc, c='k')
    ax.plot(ts, auprc_bl, c='k', ls='--')
    ax.set_ylabel('AUPRC')
    #ax.set_ylim(0,0.31)
    ax.set_yticks([0,0.1,0.2,0.3])
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.grid(True)
    sns.despine()

    ax = fig.add_subplot(gs[3], sharex=ax0)
    ax.plot(ts, np.isnan(risks).mean(axis=0)*100, c='k')
    ax.set_ylabel('Not available %')
    #ax.set_ylim(0,50.1)
    ax.set_yticks([0,10,20,30,40,50])
    ax.set_xlabel('Years before diagnosis/censoring')
    ax.grid(True)
    sns.despine()

    plt.tight_layout()
    plt.savefig(f'risk_sequence_{outcome_of_interest}_{cv_method}.png', bbox_inches='tight', dpi=300)


if __name__=='__main__':
    main()

