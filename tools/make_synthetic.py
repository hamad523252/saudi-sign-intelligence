import json, numpy as np, sys
sys.path.insert(0,"train")
from features import FEAT_DIM
rng=np.random.default_rng(1)
labels=[v["id"] for v in json.load(open('./vocab.json'))][:6]
protos={l:rng.normal(0,1,(4,FEAT_DIM)) for l in labels}
for signer in ["H001","H002"]:
    out={"signer":signer,"samples":[]}
    for l in labels:
        for k in range(25):
            n=rng.integers(40,70); t=np.linspace(0,3,n)
            P=protos[l]; fr=(P[0]+np.outer(np.sin(t),P[1])+np.outer(np.cos(2*t),P[2])*0.5)*0.3
            fr+=rng.normal(0,0.15,fr.shape); fr[:,-2:]=1
            out["samples"].append({"label":l,"frames":fr.tolist()})
    json.dump(out,open(f'data/synthetic_{signer}.json','w'))
print("ok")
