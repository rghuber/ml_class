#!/usr/bin/env python
import sys
import pubchempy as pcp


f=open("drug_pubchem_index.txt").readlines()
drug_names=[]
drug_cns=[]
drug_cid=[]
drug_smiles=[]
for i in f[1:]:
    l=i.split("[")[1].split("]")[0]
    if len(l.split()) == 1:
        drug_cid.append(int(l))
        l=i.split(",")
        drug_names.append(l[0])
        drug_cns.append(l[1])
        drug_smiles.append(l[-1].strip())

of=open("drug_properties.txt","w")
of.write("# compound name, pubchem_id, CNS, xLogP, TPSA, SMILES\n")
for i in range(len(drug_names)):
    props=pcp.Compound.from_cid(drug_cid[i]).to_dict()
    try:
        tpsa=float(props['tpsa'])
        xlogp=float(props['xlogp'])
        of.write("%20s, %10d, %10s, %8.3f, %10.3f, %s\n"%(drug_names[i],drug_cid[i],drug_cns[i],xlogp,tpsa,drug_smiles[i]))
    except:
        pass
of.close()
