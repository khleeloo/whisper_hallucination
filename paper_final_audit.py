#!/usr/bin/env python3
"""Final paper-facing audit for mismatch, cross-model stress, decision utility, and CIs.

This script uses cached outputs only: no ASR/LM inference is rerun. It deliberately
implements the final WER normalizer locally so the historical optional-apostrophe
`n't` bug cannot leak into paper-facing numbers.
"""
from __future__ import annotations

import json, math, re, unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path('/scratch/vemotionsys/rmfrieske/whisper_hallucination')
OUT = ROOT / 'paper_final_audit'
CONDITIONS = ['none','full_noise_amp0.5_dur0.0','full_noise_amp0.75_dur0.0']
INVALID = {'', 'undefined', 'none', 'null', 'nan', 'n/a', 'na'}
SPECIAL = re.compile(r'<\|[^|]+\|>')
STRICT_WER = 0.5
SEED = 20260824
BOOT = 2000


def norm(text: object) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ''
    s = unicodedata.normalize('NFKC', str(text)).lower()
    s = SPECIAL.sub(' ', s)
    s = s.replace('’', "'").replace('‘', "'").replace('`', "'")
    s = re.sub(r"\bwon't\b", 'will not', s)
    s = re.sub(r"\bcan't\b", 'can not', s)
    s = re.sub(r"\bshan't\b", 'shall not', s)
    # IMPORTANT: apostrophe is required. Never use n['’]?t here.
    s = re.sub(r"n't\b", ' not', s)
    s = re.sub(r"'ll\b", ' will', s)
    s = re.sub(r"'re\b", ' are', s)
    s = re.sub(r"'ve\b", ' have', s)
    s = re.sub(r"'m\b", ' am', s)
    s = re.sub(r'[-‐‑‒–—―/]+', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s, flags=re.UNICODE).replace('_',' ')
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r"\b(i|you|he|she|it|we|they)\s+ll\b", r'\1 will', s)
    s = re.sub(r"\b(you|we|they)\s+re\b", r'\1 are', s)
    s = re.sub(r"\b(i|you|we|they)\s+ve\b", r'\1 have', s)
    s = re.sub(r"\bi\s+m\b", 'i am', s)
    s = re.sub(r"\b(can|will)\s+n\s+t\b", r'\1 not', s)
    s = re.sub(r"\b(\w+)\s+n\s+t\b", r'\1 not', s)
    return re.sub(r'\s+', ' ', s).strip()


def lev(a: Sequence[str], b: Sequence[str]) -> int:
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):
            cur.append(min(cur[-1]+1, prev[j]+1, prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]


def wer(ref: object, hyp: object) -> float:
    r,h=norm(ref),norm(hyp)
    if r in INVALID: return float('nan')
    rw,hw=r.split(),h.split()
    return lev(rw,hw)/len(rw)


def safety_asserts() -> None:
    assert norm('want') == 'want'
    assert norm('instrument') == 'instrument'
    assert norm('establishment') == 'establishment'
    assert norm("doesn't") == 'does not'
    assert wer("She'll go", 'she ll go') == 0.0


def pick_col(df: pd.DataFrame, names: Sequence[str]) -> str:
    for n in names:
        if n in df.columns: return n
    raise KeyError(f'None of {list(names)} found. Columns={list(df.columns)}')


def add_final_wer(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy()
    rc=pick_col(out,['reference','ref','target','sentence'])
    hc=pick_col(out,['hypothesis','hyp','prediction','transcription'])
    old='WER' if 'WER' in out.columns else ('wer' if 'wer' in out.columns else None)
    if old: out['WER_previous_finalaudit']=pd.to_numeric(out[old],errors='coerce')
    out['reference_norm_final']=out[rc].map(norm)
    out['hypothesis_norm_final']=out[hc].map(norm)
    out['valid_reference_final']=~out['reference_norm_final'].isin(INVALID)
    out['WER_final']=[wer(r,h) for r,h in zip(out[rc],out[hc])]
    return out


def rep34_series(df: pd.DataFrame) -> pd.Series:
    if 'rep34' in df.columns: return df['rep34'].astype(bool)
    if {'rep3','rep4'}.issubset(df.columns):
        return (pd.to_numeric(df.rep3,errors='coerce').fillna(0)>0)|(pd.to_numeric(df.rep4,errors='coerce').fillna(0)>0)
    if {'trigram_rep_count','fourgram_rep_count'}.issubset(df.columns):
        return (pd.to_numeric(df.trigram_rep_count,errors='coerce').fillna(0)>0)|(pd.to_numeric(df.fourgram_rep_count,errors='coerce').fillna(0)>0)
    hc=pick_col(df,['hypothesis','hyp','prediction','transcription'])
    vals=[]
    for x in df[hc]:
        t=norm(x).split(); hit=False
        for n in (3,4):
            if len(t)>=n:
                c=Counter(tuple(t[i:i+n]) for i in range(len(t)-n+1))
                if any(v>1 for v in c.values()): hit=True
        vals.append(hit)
    return pd.Series(vals,index=df.index)


def lm_cols(df: pd.DataFrame) -> Tuple[str,str]:
    q=pick_col(df,['qwen_plaus','normalized_sentence_score_Qwen3-0.6B','qwen_plausibility'])
    g=pick_col(df,['gpt2_plaus','normalized_sentence_score_gpt2','gpt2_plausibility'])
    return q,g


def derive_thresholds(df: pd.DataFrame) -> Dict[str,float]:
    q,g=lm_cols(df)
    x=df.copy()
    if 'split' in x.columns:
        clean=x[(x['split'].astype(str)=='dev') & (x['perturbation'].astype(str)=='none')]
    else:
        clean=x[x['perturbation'].astype(str)=='none'] if 'perturbation' in x.columns else x
    clean=clean[clean.valid_reference_final & np.isfinite(clean.WER_final)]
    if clean.empty: raise ValueError('No clean rows for thresholds')
    return {'wer':float(clean.WER_final.mean()),'qwen':float(pd.to_numeric(clean[q]).mean()),'gpt2':float(pd.to_numeric(clean[g]).mean()),'N':int(len(clean))}


def label(df: pd.DataFrame, th: Dict[str,float]) -> pd.DataFrame:
    out=df.copy(); q,g=lm_cols(out)
    valid=out.valid_reference_final & np.isfinite(out.WER_final)
    diag=valid & (out.WER_final>th['wer']); strict=valid & (out.WER_final>STRICT_WER)
    out['diag_h_qwen_final']=diag & (pd.to_numeric(out[q])>th['qwen'])
    out['diag_h_gpt2_final']=diag & (pd.to_numeric(out[g])>th['gpt2'])
    out['strict_h_qwen_final']=strict & (pd.to_numeric(out[q])>th['qwen'])
    out['strict_h_gpt2_final']=strict & (pd.to_numeric(out[g])>th['gpt2'])
    out['strict_h_union_final']=out.strict_h_qwen_final|out.strict_h_gpt2_final
    out['rep34_final']=rep34_series(out)
    return out


def concentration(vals: Iterable[object]) -> Tuple[float,float]:
    z=[norm(x) for x in vals]; z=[x for x in z if x]
    if not z: return 0.,0.
    c=pd.Series(z).value_counts(); return float(c.iloc[0]/len(z)),float(c.iloc[:10].sum()/len(z))


def summarize_model(df: pd.DataFrame, model: str) -> pd.DataFrame:
    q,g=lm_cols(df); hc=pick_col(df,['hypothesis','hyp','prediction','transcription'])
    x=df[df['split'].astype(str)=='test'] if 'split' in df.columns else df
    rows=[]
    for cond in CONDITIONS:
        z=x[x.perturbation.astype(str)==cond]
        if z.empty: continue
        zvalid=z[z.valid_reference_final]
        t1,t10=concentration(z[hc])
        rows.append({'model':model,'condition':cond,'N':len(z),'WER':zvalid.WER_final.mean(),
          'qwen_plaus':pd.to_numeric(z[q]).mean(),'gpt2_plaus':pd.to_numeric(z[g]).mean(),
          'diag_H_qwen_pct':100*z.diag_h_qwen_final.mean(),'diag_H_gpt2_pct':100*z.diag_h_gpt2_final.mean(),
          'strict_H_qwen_pct':100*z.strict_h_qwen_final.mean(),'strict_H_gpt2_pct':100*z.strict_h_gpt2_final.mean(),
          'strict_H_union_pct':100*z.strict_h_union_final.mean(),'rep34_pct':100*z.rep34_final.mean(),
          'top1_mass_pct':100*t1,'top10_mass_pct':100*t10})
    return pd.DataFrame(rows)


def bootstrap(df: pd.DataFrame, model: str, B: int=BOOT) -> pd.DataFrame:
    q,_=lm_cols(df); hc=pick_col(df,['hypothesis','hyp','prediction','transcription'])
    x=df[df['split'].astype(str)=='test'] if 'split' in df.columns else df
    rng=np.random.default_rng(SEED); rows=[]
    for cond in CONDITIONS:
        z=x[x.perturbation.astype(str)==cond].reset_index(drop=True)
        if z.empty: continue
        n=len(z); vals={k:[] for k in ['WER','qwen_plaus','diag_H_qwen_pct','strict_H_qwen_pct','rep34_pct','top1_mass_pct']}
        for _ in range(B):
            s=z.iloc[rng.integers(0,n,n)]
            v=s[s.valid_reference_final]
            vals['WER'].append(float(v.WER_final.mean()))
            vals['qwen_plaus'].append(float(pd.to_numeric(s[q]).mean()))
            vals['diag_H_qwen_pct'].append(100*float(s.diag_h_qwen_final.mean()))
            vals['strict_H_qwen_pct'].append(100*float(s.strict_h_qwen_final.mean()))
            vals['rep34_pct'].append(100*float(s.rep34_final.mean()))
            vals['top1_mass_pct'].append(100*concentration(s[hc])[0])
        for metric,a in vals.items():
            lo,hi=np.percentile(a,[2.5,97.5]); rows.append({'model':model,'condition':cond,'metric':metric,'ci_low':lo,'ci_high':hi,'B':B})
    return pd.DataFrame(rows)


def load_and_finalize(path: Path) -> Tuple[pd.DataFrame,Dict[str,float]]:
    d=add_final_wer(pd.read_csv(path)); th=derive_thresholds(d); return label(d,th),th


def mismatch_audit() -> None:
    base=ROOT/'eval_validation/per_utterance_base_ckpt14000.csv'; ed=ROOT/'eval_64pct'
    files={'Base':[base],'RR':[ed/'per_utterance_rr_64pct_checkpoint-9375.csv'],'RU':[ed/'per_utterance_ru_64pct_checkpoint-9375.csv'],
      'UR':[ed/'per_utterance_ur_64pct_checkpoint-10000_shard00-of-02.csv',ed/'per_utterance_ur_64pct_checkpoint-10000_shard01-of-02.csv'],
      'UU':[ed/'per_utterance_uu_64pct_final_shard00-of-02.csv',ed/'per_utterance_uu_64pct_final_shard01-of-02.csv']}
    ds={k:add_final_wer(pd.concat([pd.read_csv(p) for p in ps],ignore_index=True)) for k,ps in files.items()}
    q0,g0=lm_cols(ds['Base']); th={'wer':float(ds['Base'].WER_final.mean()),'qwen':float(pd.to_numeric(ds['Base'][q0]).mean()),'gpt2':float(pd.to_numeric(ds['Base'][g0]).mean())}
    rows=[]; robust=[]; changed=[]
    for k,d in ds.items():
        q,g=lm_cols(d); rep=rep34_series(d); valid=d.valid_reference_final & np.isfinite(d.WER_final)
        diagq=valid&(d.WER_final>th['wer'])&(pd.to_numeric(d[q])>th['qwen'])
        strictq=valid&(d.WER_final>.5)&(pd.to_numeric(d[q])>th['qwen'])
        rows.append({'condition':k,'N':len(d),'WER_pct':100*d.WER_final.mean(),'LM_Qwen':pd.to_numeric(d[q]).mean(),
          'diag_H_Qwen_pct':100*diagq.mean(),'strict_H_Qwen_pct':100*strictq.mean(),
          'Rep3_pct':100*(pd.to_numeric(d.get('trigram_rep_count',0),errors='coerce').fillna(0)>0).mean() if 'trigram_rep_count' in d else np.nan,
          'Rep4_pct':100*(pd.to_numeric(d.get('fourgram_rep_count',0),errors='coerce').fillna(0)>0).mean() if 'fourgram_rep_count' in d else np.nan})
        for qt in [None,.6,.7]:
            qthr=th['qwen'] if qt is None else qt
            m=valid&(d.WER_final>.5)&(pd.to_numeric(d[q])>qthr)&(~rep)
            robust.append({'criterion':f'WER>0.5,Qwen>{qthr:.4f}' if qt is None else f'WER>0.5,Qwen>{qt:.1f}','condition':k,'rate_pct':100*m.mean(),'count':int(m.sum())})
        if 'WER_previous_finalaudit' in d:
            m=np.isfinite(d.WER_previous_finalaudit)&np.isfinite(d.WER_final)&(np.abs(d.WER_previous_finalaudit-d.WER_final)>1e-12)
            if m.any():
                z=d.loc[m,[c for c in ['reference','hypothesis','WER_previous_finalaudit','WER_final'] if c in d]].copy(); z['condition']=k; changed.append(z)
    pd.DataFrame(rows).to_csv(OUT/'mismatch_summary_corrected.csv',index=False)
    pd.DataFrame(robust).to_csv(OUT/'mismatch_robustness_corrected.csv',index=False)
    if changed: pd.concat(changed,ignore_index=True).to_csv(OUT/'mismatch_rows_changed.csv',index=False)
    (OUT/'mismatch_thresholds_final.json').write_text(json.dumps(th,indent=2)+'\n')


def collapse_lexicon(baseline: pd.DataFrame,min_cov=.99,cap=20) -> set[str]:
    dev=baseline[baseline.split.astype(str)=='dev'].copy(); dev['hn']=dev[pick_col(dev,['hypothesis'])].map(norm)
    stress=dev[dev.perturbation.astype(str)!='none']; clean=dev[dev.perturbation.astype(str)=='none']
    counts=stress[stress.hn!=''].hn.value_counts(); lex=set()
    for h in counts.index:
        if len(lex)>=cap: break
        trial=lex|{h}
        if 1-clean.hn.isin(trial).mean()>=min_cov-1e-12: lex=trial
    return lex


def decision_audit(model: str, base_path: Path, outdir: Path) -> pd.DataFrame:
    base,th=load_and_finalize(base_path)
    anti=add_final_wer(pd.read_csv(outdir/'antirep_scored_test.csv'))
    # Keep cached LM scores; thresholds come from final-rescored baseline DEV.
    anti=label(anti,th)
    bs=summarize_model(base,model); an=summarize_model(anti,model)
    lex=collapse_lexicon(base); test=base[base.split.astype(str)=='test'].copy(); test['hn']=test[pick_col(test,['hypothesis'])].map(norm)
    rows=[]
    for cond in CONDITIONS:
        b=bs[bs.condition==cond].iloc[0]; a=an[an.condition==cond].iloc[0]
        z=test[test.perturbation.astype(str)==cond]; rej=z.hn.isin(lex); h=z.strict_h_qwen_final
        rows.append({'model':model,'condition':cond,'WER_baseline':b.WER,'WER_antirep':a.WER,'delta_WER':a.WER-b.WER,
          'strict_H_qwen_baseline_pct':b.strict_H_qwen_pct,'strict_H_qwen_antirep_pct':a.strict_H_qwen_pct,'delta_strict_H_qwen_pp':a.strict_H_qwen_pct-b.strict_H_qwen_pct,
          'rep34_baseline_pct':b.rep34_pct,'rep34_antirep_pct':a.rep34_pct,'delta_rep34_pp':a.rep34_pct-b.rep34_pct,
          'top1_baseline_pct':b.top1_mass_pct,'top1_antirep_pct':a.top1_mass_pct,'delta_top1_pp':a.top1_mass_pct-b.top1_mass_pct,
          'collapse_abstention_pct':100*rej.mean(),'collapse_strict_H_qwen_capture_pct':100*((rej&h).sum()/h.sum() if h.sum() else 0),'lexicon_size':len(lex)})
    return pd.DataFrame(rows)


def consistency_and_bootstrap() -> None:
    sources={
      'Raw Whisper':ROOT/'pretrained_whisper_stress_pipeline/rescore_explore/scored_outputs_corrected.csv',
      'Adapted Whisper':ROOT/'clean_wer_rescore/scored_outputs_cleanwer.csv',
      'SeamlessM4T-v2':ROOT/'seamless_m4t_v2_stress_pipeline_fixedwer/scored_outputs.csv'}
    sums=[]; cis=[]; thresholds={}
    for name,p in sources.items():
        if not p.exists():
            print(f'WARNING missing cross-model source: {p}'); continue
        d,th=load_and_finalize(p); thresholds[name]=th; sums.append(summarize_model(d,name)); cis.append(bootstrap(d,name))
    summary=pd.concat(sums,ignore_index=True); summary.to_csv(OUT/'cross_model_summary_verified.csv',index=False)
    pd.concat(cis,ignore_index=True).to_csv(OUT/'cross_model_bootstrap95.csv',index=False)
    (OUT/'cross_model_thresholds_final.json').write_text(json.dumps(thresholds,indent=2)+'\n')
    # Explicit stale-vs-final adapted audit.
    legacy=ROOT/'acoustic_stress_full/per_utterance_acoustic_stress.csv'
    rows=[]
    if legacy.exists():
        old=pd.read_csv(legacy); oldwer='WER' if 'WER' in old else ('wer' if 'wer' in old else None)
        for c in CONDITIONS:
            z=old[old.perturbation.astype(str)==c]
            if len(z) and oldwer: rows.append({'source':'legacy acoustic_stress_full','condition':c,'WER_as_stored':pd.to_numeric(z[oldwer]).mean()})
    fin=summary[summary.model=='Adapted Whisper']
    for _,r in fin.iterrows(): rows.append({'source':'final corrected adapted source','condition':r.condition,'WER_as_stored':r.WER})
    pd.DataFrame(rows).to_csv(OUT/'adapted_whisper_consistency_audit.csv',index=False)


def main() -> None:
    safety_asserts(); OUT.mkdir(parents=True,exist_ok=True)
    mismatch_audit()
    consistency_and_bootstrap()
    decisions=[]
    decisions.append(decision_audit('Raw Whisper',ROOT/'pretrained_whisper_stress_pipeline/rescore_explore/scored_outputs_corrected.csv',ROOT/'decision_utility_raw_whisper'))
    decisions.append(decision_audit('SeamlessM4T-v2',ROOT/'seamless_m4t_v2_stress_pipeline_fixedwer/scored_outputs.csv',ROOT/'decision_utility_seamless'))
    pd.concat(decisions,ignore_index=True).to_csv(OUT/'decision_utility_final_rescore.csv',index=False)
    report=[
      'FINAL PAPER AUDIT COMPLETE',
      '1 mismatch rescore: mismatch_summary_corrected.csv + mismatch_robustness_corrected.csv',
      '2 consistency: cross_model_summary_verified.csv + adapted_whisper_consistency_audit.csv',
      '3 decision utility final rescore: decision_utility_final_rescore.csv',
      '5 bootstrap 95% CIs: cross_model_bootstrap95.csv',
      f'bootstrap replicates={BOOT}, seed={SEED}',
      'WER normalizer safety checks: PASS (want/instrument/establishment preserved; contractions normalized).']
    (OUT/'README_results.txt').write_text('\n'.join(report)+'\n')
    print('\n'.join(report))
    print(f'Outputs: {OUT}')

if __name__=='__main__': main()
