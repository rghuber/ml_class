#!/usr/bin/env python
import sys
import pubchempy as pcp


f=open("drugs.txt").readlines()
drug_names=[]
drug_smiles=[]
drug_cns=[]
for i in f[1:]:
    l=i.split()
    if len(l) > 2:
        drug_names.append(l[0])
        drug_cns.append(l[1])
        drug_smiles.append(l[2].strip())

drug_cids=[]
for i in drug_names:
    cid_list=pcp.get_cids(i,'name','compound', list_return='flat')
    print "%20s %5d hits"%(i,len(cid_list))
    drug_cids.append(cid_list)

of=open("drug_pubchem_index.txt","w")
for i in range(len(drug_names)):
    of.write("%20s, %6s, %s, %s\n"%(drug_names[i],drug_cns[i],drug_cids[i],drug_smiles[i]))
of.close()

