#!/usr/bin/env python3
import argparse,csv,hashlib,json,math,statistics
FIELDS=('w11','w21','w12','w22')

def load(path):
    rows=list(csv.DictReader(open(path,newline='')))
    chains={}
    for row in rows:
        cid=row['chain']
        chains.setdefault(cid,[]).append({k:float(row[k]) for k in FIELDS})
    return rows,chains

def estimate(rows):
    means={k:statistics.mean(r[k] for r in rows) for k in FIELDS}
    ratio=means['w22']*means['w11']/(means['w21']*means['w12'])
    if not ratio>0: raise ValueError('non-positive Creutz ratio argument')
    return {'means':means,'ratio':ratio,'creutz_22':-math.log(ratio)}

def blocked_jackknife(chains,block_size):
    if block_size<=0: raise ValueError('invalid block size')
    for cid,seq in chains.items():
        if len(seq)%block_size: raise ValueError(f'partial block in chain {cid}')
    reps=[]
    for cid,seq in chains.items():
        for start in range(0,len(seq),block_size):
            kept=[]
            for other,seq2 in chains.items():
                for i,row in enumerate(seq2):
                    if other==cid and start<=i<start+block_size: continue
                    kept.append(row)
            reps.append(estimate(kept)['creutz_22'])
    n=len(reps); mean=statistics.mean(reps)
    se=math.sqrt((n-1)/n*sum((x-mean)**2 for x in reps))
    return {'block_size':block_size,'replicate_count':n,'replicate_mean':mean,'standard_error':se,'replicate_min':min(reps),'replicate_max':max(reps),'replicates':reps}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('history'); args=ap.parse_args()
    raw=open(args.history,'rb').read(); rows,chains=load(args.history)
    base=estimate([r for seq in chains.values() for r in seq])
    scans=[blocked_jackknife(chains,b) for b in (1,2,4)]
    result={'history_sha256':hashlib.sha256(raw).hexdigest(),'chain_count':len(chains),'samples_per_chain':{k:len(v) for k,v in sorted(chains.items())},'central':base,'jackknife':scans}
    text=json.dumps(result,sort_keys=True,separators=(',',':')); digest=hashlib.sha256(text.encode()).hexdigest()
    expected='679ced05098d7f0f596e34eb88c714e6371eadf8e424b70959aa9d460a827ebe'
    if digest!=expected: raise AssertionError(('frozen result hash',digest,expected))
    print('ok'); print('result_sha256='+digest); print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
