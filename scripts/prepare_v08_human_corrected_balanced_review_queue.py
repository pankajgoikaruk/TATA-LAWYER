# scripts/prepare_v08_human_corrected_balanced_review_queue.py
from __future__ import annotations
import argparse, json, shutil
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd

DEFAULT_TARGET_LABELS=["Brene_Brown","Eckhart_Tolle","Eric_Thomas","Gary_Vee","Jay_Shetty","Nick_Vujicic"]
DEFAULT_EVENT_LABELS=["music_present","audience_reaction_present","silence_present"]
DEFAULT_NONTARGET=["Les_Brown","Mel_Robbins","Oprah_Winfrey","Rabin_Sharma","Simon_Sinek"]

def now(): return datetime.now().isoformat(timespec='seconds')
def read_json(p:Path)->dict[str,Any]: return json.loads(p.read_text(encoding='utf-8'))
def load_labels(p:Path)->list[str]:
    obj=read_json(p); labs=obj.get('labels', obj) if isinstance(obj,dict) else obj
    labs=[str(x) for x in labs]
    if not labs: raise RuntimeError(f'No labels found in {p}')
    return labs

def load_groups(labels:list[str], cfg_path:Path|None):
    cfg=read_json(cfg_path) if cfg_path and cfg_path.exists() else {}
    groups=cfg.get('label_groups',{}) if isinstance(cfg,dict) else {}
    src=cfg.get('source_matching',{}) if isinstance(cfg,dict) else {}
    target=[x for x in groups.get('target_labels',DEFAULT_TARGET_LABELS) if x in labels]
    open_label=str(groups.get('open_set_label','other_speaker_present'))
    events=[x for x in groups.get('event_labels',DEFAULT_EVENT_LABELS) if x in labels]
    non_target=[str(x) for x in src.get('known_non_target_source_classes',DEFAULT_NONTARGET)] or DEFAULT_NONTARGET
    missing=[x for x in target+events+[open_label] if x not in labels]
    if missing: raise RuntimeError(f'Configured labels missing from labels_json: {missing}')
    return target, open_label, events, non_target

def read_csv(p:Path)->pd.DataFrame:
    if not p.exists():
        print(f'[WARN] Missing file: {p}')
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False)

def ensure_labels(df:pd.DataFrame, labels:list[str])->pd.DataFrame:
    df=df.copy()
    for lab in labels:
        if lab not in df.columns: df[lab]=0
        df[lab]=pd.to_numeric(df[lab], errors='coerce').fillna(0).astype(int)
    return df

def refresh(df:pd.DataFrame, labels:list[str])->pd.DataFrame:
    if len(df)==0: return ensure_labels(df, labels)
    df=ensure_labels(df, labels)
    df['manual_labels']=df.apply(lambda r:'|'.join([lab for lab in labels if int(r.get(lab,0))==1]), axis=1)
    df['labels']=df['manual_labels']
    df['num_active_labels']=df[labels].sum(axis=1).astype(int)
    return df

def non_target_mask(df:pd.DataFrame, non_target:list[str])->pd.Series:
    if len(df)==0: return pd.Series(dtype=bool)
    cols=[c for c in ['source_class_dir','source_file','source_path','source_rel_path','parent_clip_id'] if c in df.columns]
    if not cols: return pd.Series([False]*len(df), index=df.index)
    text=df[cols].astype(str).agg(' '.join, axis=1)
    mask=pd.Series([False]*len(df), index=df.index)
    for cls in non_target: mask |= text.str.contains(cls, regex=False, na=False)
    return mask

def add_review(df:pd.DataFrame, *, source_tag:str, priority:str, task:str, role:str, instruction:str)->pd.DataFrame:
    df=df.copy()
    front={'review_source_file':source_tag,'review_priority':priority,'review_task':task,'dataset_role':role,'human_review_status':'pending','human_review_notes':'','review_instruction':instruction}
    for k,v in reversed(list(front.items())):
        if k in df.columns: df[k]=v
        else: df.insert(0,k,v)
    return df

def force_nontarget(df:pd.DataFrame, labels:list[str], target:list[str], open_label:str, non_target:list[str])->pd.DataFrame:
    df=ensure_labels(df, labels); m=non_target_mask(df, non_target)
    if len(df) and m.any():
        for lab in target: df.loc[m,lab]=0
        df.loc[m,open_label]=1
    return refresh(df, labels)

def sort_queue(df:pd.DataFrame, events:list[str])->pd.DataFrame:
    if len(df)==0: return df
    score=pd.Series([0.0]*len(df), index=df.index)
    for lab in events:
        for col in [f'lawyer_score_{lab}',f'lawyer_seg_max_{lab}',f'lawyer_seg_mean_{lab}',f'parent_prob_{lab}',f'parent_pred_{lab}',lab]:
            if col in df.columns: score += pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    df=df.copy(); df['_review_sort_score']=score
    sort_cols=[c for c in ['review_priority','_review_sort_score','source_class_dir','parent_clip_id'] if c in df.columns]
    if sort_cols: df=df.sort_values(sort_cols, ascending=[c!='_review_sort_score' for c in sort_cols])
    return df.drop(columns=['_review_sort_score'], errors='ignore')

def save_queue(df:pd.DataFrame, path:Path, labels:list[str], events:list[str])->dict[str,Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    df=sort_queue(refresh(df, labels), events)
    df.to_csv(path, index=False)
    return {'file':str(path),'rows':int(len(df)),'label_counts':{lab:int(pd.to_numeric(df[lab],errors='coerce').fillna(0).astype(int).sum()) for lab in labels if lab in df.columns}}

def combine(dfs:list[pd.DataFrame], labels:list[str])->pd.DataFrame:
    dfs=[d for d in dfs if d is not None and len(d)>0]
    if not dfs: return refresh(pd.DataFrame(), labels)
    out=pd.concat(dfs, ignore_index=True, sort=False)
    if 'dataset_role' in out.columns and 'parent_clip_id' in out.columns:
        out['_key']=out['dataset_role'].astype(str)+'::'+out['parent_clip_id'].astype(str)
        out=out.drop_duplicates('_key', keep='first').drop(columns=['_key'])
    return refresh(out, labels)

def copy_reference(lawyer_dir:Path, out_root:Path)->dict[str,Any]:
    dest=out_root/'frozen_reference'/'lawyer_v08_auto'; dest.mkdir(parents=True, exist_ok=True)
    copied=[]
    if lawyer_dir.exists():
        for p in lawyer_dir.iterdir():
            if p.is_file() and p.suffix.lower() in {'.csv','.json','.md'}:
                t=dest/p.name; shutil.copy2(p,t); copied.append(str(t))
    return {'source_lawyer_dir':str(lawyer_dir),'frozen_reference_dir':str(dest),'num_copied_files':len(copied),'copied_files':copied}

def write_readme(path:Path, labels:list[str], target:list[str], open_label:str, events:list[str], non_target:list[str]):
    target_lines='\n'.join([x+' = 0' for x in target])
    txt=f"""# v0.8-human-corrected-balanced Manual Review Queue

Generated: `{now()}`

Current v0.8 automatic LAWYER outputs are frozen under:

```text
frozen_reference/lawyer_v08_auto/
```

Do **not** edit frozen reference files.

## Files to edit

```text
01_lawyer_needs_review_FULL.csv
02_raw_hybrid_nontarget_context_review.csv
03_holdout_nontarget_context_check.csv
04_lawyer_event_positive_review.csv
05_lawyer_nontarget_context_review.csv
06_v08_training_review_master_deduplicated.csv
07_holdout_context_review_master.csv
```

## Non-target rule

Known non-target source classes:

```text
{chr(10).join(non_target)}
```

For these rows, keep:

```text
{target_lines}
{open_label} = 1
```

Only manually check/edit these background/event labels:

```text
{chr(10).join(events)}
```

## What to edit

For each reviewed row, update label columns as `0` or `1`, then update:

```text
manual_labels
labels
num_active_labels
human_review_status = reviewed
human_review_notes
```

Do not edit IDs/paths/features:

```text
parent_clip_id
sample_id
source_path
source_file
feat_relpath
split
dataset_role
review_source_file
```

## Training vs holdout

`06_v08_training_review_master_deduplicated.csv` is for training-side correction.

`07_holdout_context_review_master.csv` is holdout-only. Do not mix holdout rows into training.
"""
    path.write_text(txt, encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(description='Prepare v0.8-human-corrected-balanced manual review queues.')
    ap.add_argument('--v06_root', default='human_talk_workspace/tata_v0.6_raw_pipeline')
    ap.add_argument('--v08_root', default='human_talk_workspace/tata_v0.8_raw_pipeline')
    ap.add_argument('--out_root', default='human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline')
    ap.add_argument('--labels_json', default='configs/human_talk_10label_schema.json')
    ap.add_argument('--lawyer_config', default='configs/lawyer_v08_human_talk.json')
    ap.add_argument('--lawyer_dir', default='')
    args=ap.parse_args()
    v06=Path(args.v06_root); v08=Path(args.v08_root); out=Path(args.out_root)
    lawyer_dir=Path(args.lawyer_dir) if args.lawyer_dir else v08/'raw_tata_pseudo_routing'/'lawyer_v08'
    labels=load_labels(Path(args.labels_json)); target, open_label, events, non_target=load_groups(labels, Path(args.lawyer_config))
    qdir=out/'manual_review_queue'; qdir.mkdir(parents=True, exist_ok=True)
    frozen=copy_reference(lawyer_dir, out)
    paths={'lawyer_all':lawyer_dir/'lawyer_v08_parent_labels_all.csv','lawyer_needs':lawyer_dir/'lawyer_v08_needs_review.csv','raw_hybrid_reviewed':v06/'manual_review_queue'/'02_raw_hybrid_needs_review_MANUAL_CORRECTION_FINAL_refreshed.csv','holdout_gt':v06/'manual_review_queue'/'01_raw_final_holdout_GROUND_TRUTH_FINAL_refreshed.csv'}
    lawyer_all=refresh(read_csv(paths['lawyer_all']), labels); lawyer_needs=refresh(read_csv(paths['lawyer_needs']), labels); raw_hybrid=refresh(read_csv(paths['raw_hybrid_reviewed']), labels); holdout=refresh(read_csv(paths['holdout_gt']), labels)
    q1=add_review(lawyer_needs, source_tag='lawyer_v08_needs_review.csv', priority='P1', task='lawyer_uncertain_full_label_review', role='training_candidate', instruction='Review all labels because LAWYER routed this parent clip to needs_review.')
    raw_nt=force_nontarget(raw_hybrid[non_target_mask(raw_hybrid, non_target)].copy(), labels, target, open_label, non_target)
    q2=add_review(raw_nt, source_tag='02_raw_hybrid_needs_review_MANUAL_CORRECTION_FINAL_refreshed.csv', priority='P2', task='raw_hybrid_non_target_context_review', role='training_candidate', instruction=f'Keep target speakers=0 and {open_label}=1. Check only: {", ".join(events)}.')
    hold_nt=force_nontarget(holdout[non_target_mask(holdout, non_target)].copy(), labels, target, open_label, non_target)
    q3=add_review(hold_nt, source_tag='01_raw_final_holdout_GROUND_TRUTH_FINAL_refreshed.csv', priority='P1_HOLDOUT', task='holdout_non_target_context_ground_truth_check', role='holdout_ground_truth', instruction=f'HOLDOUT ONLY. Do not use for training. Keep target speakers=0 and {open_label}=1. Check: {", ".join(events)}.')
    event_mask=pd.Series([False]*len(lawyer_all), index=lawyer_all.index)
    for lab in events: event_mask |= pd.to_numeric(lawyer_all[lab], errors='coerce').fillna(0).astype(int).eq(1)
    q4=add_review(lawyer_all[event_mask].copy(), source_tag='lawyer_v08_parent_labels_all.csv', priority='P3', task='lawyer_event_positive_context_review', role='training_candidate', instruction=f'Review event/context labels: {", ".join(events)}. Speaker labels usually need only spot-checking.')
    law_nt=force_nontarget(lawyer_all[non_target_mask(lawyer_all, non_target)].copy(), labels, target, open_label, non_target)
    q5=add_review(law_nt, source_tag='lawyer_v08_parent_labels_all.csv', priority='P2', task='lawyer_non_target_context_review', role='training_candidate', instruction=f'Keep target speakers=0 and {open_label}=1. Check event labels: {", ".join(events)}.')
    outputs={}
    for name,df in [('01_lawyer_needs_review_FULL.csv',q1),('02_raw_hybrid_nontarget_context_review.csv',q2),('03_holdout_nontarget_context_check.csv',q3),('04_lawyer_event_positive_review.csv',q4),('05_lawyer_nontarget_context_review.csv',q5)]:
        outputs[name]=save_queue(df, qdir/name, labels, events)
    outputs['06_v08_training_review_master_deduplicated.csv']=save_queue(combine([q1,q2,q4,q5], labels), qdir/'06_v08_training_review_master_deduplicated.csv', labels, events)
    outputs['07_holdout_context_review_master.csv']=save_queue(combine([q3], labels), qdir/'07_holdout_context_review_master.csv', labels, events)
    summary={'generated_at':now(),'experiment':'v0.8-human-corrected-balanced','manual_review_queue':str(qdir),'labels':labels,'target_labels':target,'open_set_label':open_label,'event_labels':events,'known_non_target_source_classes':non_target,'frozen_reference':frozen,'input_files':{k:str(v) for k,v in paths.items()},'outputs':outputs}
    (qdir/'00_review_queue_summary.json').write_text(json.dumps(summary,indent=2), encoding='utf-8')
    write_readme(qdir/'00_README_REVIEW_INSTRUCTIONS.md', labels, target, open_label, events, non_target)
    print('\nv0.8-human-corrected-balanced review queues prepared'); print('-'*90); print(f'Manual review queue: {qdir}'); print(f'Frozen reference:    {frozen["frozen_reference_dir"]}\n')
    for name,info in outputs.items(): print(f'{name}: rows={info["rows"]}')
    print(f'\nSummary: {qdir/"00_review_queue_summary.json"}'); print(f'Instructions: {qdir/"00_README_REVIEW_INSTRUCTIONS.md"}')
if __name__=='__main__': main()
