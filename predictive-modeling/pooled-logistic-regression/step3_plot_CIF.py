import pickle, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import seaborn as sns
sns.set_style('ticks')
from scipy.interpolate import interp1d



def observed_cif_from_long(sids, Y, T, alpha=0.05, eps=1e-12):
    """
    KM-based observed CIF with pointwise (1-alpha) CI.
    
    Inputs
    ------
    sids : array-like, length n_row
    Y    : array-like, length n_row (0,...,0 or 0,...,0,1 per subject)
    T    : array-like, length n_row (time per row; assumed increasing within subject)
    alpha: float, CI level (default 0.05 => 95% CI)
    eps  : small float to avoid log(0)

    Returns
    -------
    event_times : array, times where KM updates (event times)
    cif         : array, CIF = 1 - S_hat(t) at event_times
    cif_lo      : array, lower CI for CIF at event_times
    cif_hi      : array, upper CI for CIF at event_times
    n_risk      : array, risk set size at each event_time
    n_event     : array, events at each event_time
    time2event  : array, subject-level last observed time
    event       : array, subject-level event indicator at last time
    """
    # ----- Extract subject-level (T_i, E_i) from long format -----
    sid2ids = pd.DataFrame(data={'sid': sids}).groupby('sid').indices
    unique_sids = np.array(sorted(sid2ids.keys()))

    # Assumption: last row for each sid corresponds to final observed time,
    # and its Y indicates whether event occurred (1) or not (0) at that final time.
    time2event = np.array([T[sid2ids[sid][-1]] for sid in unique_sids], dtype=float)
    event = np.array([Y[sid2ids[sid][-1]] for sid in unique_sids], dtype=int)

    # ----- KM times are event times -----
    event_times = np.sort(np.unique(time2event[event == 1]))

    z = norm.ppf(1 - alpha / 2)

    S = 1.0
    surv = []
    n_risk = []
    n_event = []

    # Greenwood cumulative sum term: sum d / (n*(n-d))
    greenwood_cum = 0.0
    varS = []

    for t0 in event_times:
        at_risk = time2event >= t0
        n0 = int(at_risk.sum())
        d0 = int(((time2event == t0) & (event == 1)).sum())

        # KM step
        S *= (1.0 - d0 / n0)

        surv.append(S)
        n_risk.append(n0)
        n_event.append(d0)

        # Greenwood increment (guard against n0 == d0)
        denom = n0 * (n0 - d0)
        if denom > 0:
            greenwood_cum += d0 / denom
        # Var(S) = S^2 * greenwood_cum
        varS.append((S ** 2) * greenwood_cum)

    surv = np.asarray(surv, dtype=float)
    varS = np.asarray(varS, dtype=float)

    # ----- Compute survival CI -----
    # Log-log transformed CI (recommended)
    # theta = log(-log(S)); se(theta) ~ se(S)/(S*|log(S)|)
    S_clip = np.clip(surv, eps, 1.0 - eps)
    seS = np.sqrt(np.maximum(varS, 0.0))

    denom = S_clip * np.abs(np.log(S_clip))
    denom = np.maximum(denom, eps)
    se_theta = seS / denom

    theta = np.log(-np.log(S_clip))
    theta_lo = theta - z * se_theta
    theta_hi = theta + z * se_theta

    # Convert back: S = exp(-exp(theta))
    surv_lo = np.exp(-np.exp(theta_hi))  # note inversion
    surv_hi = np.exp(-np.exp(theta_lo))  # note inversion

    surv_lo = np.clip(surv_lo, 0.0, 1.0)
    surv_hi = np.clip(surv_hi, 0.0, 1.0)

    # ----- CIF and CIF CI -----
    cif = 1.0 - surv
    # Since CIF = 1 - S, the CI inverts:
    cif_lo = 1.0 - surv_hi
    cif_hi = 1.0 - surv_lo

    n_risk = np.asarray(n_risk, dtype=int)
    n_event = np.asarray(n_event, dtype=int)

    return event_times, cif, cif_lo, cif_hi,
    #        n_risk, n_event, time2event, event)


def get_pointestimate_ci(X, nbt=1000, random_state=2026, verbose=False):
    np.random.seed(random_state)
    vals = []
    for bti in tqdm(range(nbt+1), disable=not verbose):
        if bti==0:
            Xbt = X
        else:
            btids = np.random.choice(len(X), len(X), replace=True)
            Xbt = X[btids]
        vals.append(np.nanmedian(Xbt, axis=0))
    val = vals[0]
    lb, ub = np.nanpercentile(np.array(vals[1:]), (2.5,97.5), axis=0)

    return val, lb, ub


def main():
    outcome_of_interest = sys.argv[1].strip().lower()
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
    Sp = res['S_pred_cv'][0]
    risk = res['riskX_cv'][0]
    if cv_method=='mixed':
        ids_te = res['ids_te']
        sites = sites[ids_te]
        sids  = sids[ids_te]
        X = X[ids_te]
        T = T[ids_te]
        Y = Y[ids_te]
        C = C[ids_te]
        dT = dT[ids_te]

    unique_sids=np.array(sorted(set(sids)))
    cif_t = np.arange(0,5.1,0.1)
    cifs = []
    risk_subject = []
    L = []
    for sid in unique_sids:
        ids = np.where(sids==sid)[0]
        foo = interp1d(T[ids], 1-Sp[ids], bounds_error=False)
        cifs.append(foo(cif_t))
        risk_subject.append(risk[ids[0]]) # at first visit
        L.append(T[ids[-1]])
    cifs = np.array(cifs)
    risk = np.array(risk_subject)
    L = np.array(L)
    cifs[:,0] = 0.

    #Tid = np.where(cif_t==0.5)[0][0]
    #lb, ub = np.nanpercentile(cifs[:,Tid],(100/3,200/3))
    #mask_high = cifs[:,Tid]>ub
    #mask_low = cifs[:,Tid]<lb
    lb, ub = np.nanpercentile(risk, (100/3,200/3))
    mask_high = risk>ub
    mask_low = risk<lb
    sids_high = unique_sids[mask_high]
    sids_low = unique_sids[mask_low]
    
    cifa, cifa_lb, cifa_ub = get_pointestimate_ci(cifs)
    cif1, cif1_lb, cif1_ub = get_pointestimate_ci(cifs[mask_high])
    cif0, cif0_lb, cif0_ub = get_pointestimate_ci(cifs[mask_low])

    cifa_km_t, cifa_km, cifa_km_lb, cifa_km_ub = observed_cif_from_long(sids, Y, T)
    ids = np.in1d(sids, sids_high)
    cif1_km_t, cif1_km, cif1_km_lb, cif1_km_ub = observed_cif_from_long(sids[ids], Y[ids], T[ids])
    ids = np.in1d(sids, sids_low)
    cif0_km_t, cif0_km, cif0_km_lb, cif0_km_ub = observed_cif_from_long(sids[ids], Y[ids], T[ids])

    # get NNT
    tt=np.sort(np.r_[cif0_km_t,cif1_km_t])
    foo0=interp1d(cif0_km_t,cif0_km,bounds_error=False)
    foo1=interp1d(cif1_km_t,cif1_km,bounds_error=False)
    y0=foo0(tt)
    y1=foo1(tt)
    df_nnt=pd.DataFrame(data={'Time(Year)':tt,'AbsRisk_T1':y0, 'AbsRisk_T3':y1, 'RiskDiff':y1-y0,'NNT':1/(y1-y0)}).dropna(ignore_index=True)
    df_nnt['NNT']=np.ceil(df_nnt.NNT).astype(int)
    df_nnt=df_nnt[df_nnt['Time(Year)']<=2].reset_index(drop=True)
    print(df_nnt)
    df_nnt.to_csv(f'NNT_{outcome_of_interest}_CV{cv_method}.csv', index=False)

    plt.close()
    plt.fill_between(cif_t, cif1_lb*100, cif1_ub*100,color='r',alpha=0.3)
    plt.plot(cif_t, cif1*100,c='r',lw=1.5,ls='--',label='High risk group - estimated (PLR)')
    plt.fill_between(cif1_km_t, cif1_km_lb*100, cif1_km_ub*100,color='r',alpha=0.3)
    plt.plot(cif1_km_t, cif1_km*100,c='r',label='High risk group - ground truth (KM)')
    plt.fill_between(cif_t, cif0_lb*100, cif0_ub*100,color='b',alpha=0.3)
    plt.plot(cif_t, cif0*100,c='b',lw=1.5, ls='--', label='Low risk group - estimated (PLR)')
    plt.fill_between(cif0_km_t, cif0_km_lb*100, cif0_km_ub*100,color='b',alpha=0.3)
    plt.plot(cif0_km_t, cif0_km*100,c='b',label='Low risk group - ground truth (KM)')
    plt.xlim(0,2)
    plt.ylim(0,5)
    plt.legend()
    plt.xlabel('Year since visit')
    plt.ylabel('Cumulative incidence (%)')
    plt.grid(True)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'CIF_by_risk_{outcome_of_interest}_CV{cv_method}.png', bbox_inches='tight')

    plt.close()
    plt.plot(cif_t, cifs.T*100, c='k', alpha=0.1)
    plt.xlim(0,5)
    plt.ylim(0,100)
    plt.xlabel('Year since visit')
    plt.ylabel('Cumulative incidence (%)')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'CIF_individual_{outcome_of_interest}_CV{cv_method}.png', bbox_inches='tight')

    plt.close()
    plt.fill_between(cif_t, cifa_lb*100, cifa_ub*100,color='k',alpha=0.3)
    plt.plot(cif_t, cifa*100,c='k',lw=1.5,ls='--',label='Estimated (PLR)')
    plt.fill_between(cifa_km_t, cifa_km_lb*100, cifa_km_ub*100,color='k',alpha=0.3)
    plt.plot(cifa_km_t, cifa_km*100,c='k',label='Ground truth (KM)')
    plt.xlim(0,2)
    plt.ylim(0,5)
    plt.legend()
    plt.xlabel('Year since visit')
    plt.ylabel('Cumulative incidence (%)')
    plt.grid(True)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'CIF_all_average_{outcome_of_interest}_CV{cv_method}.png', bbox_inches='tight')


if __name__=='__main__':
    main()

