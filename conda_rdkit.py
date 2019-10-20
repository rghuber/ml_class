import sys
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

f=open("drug_properties.txt").readlines()

mols=[]
names=[]
cid=[]
cns=[]
xlogp=[]
tpsa=[]

nms=[x[0] for x in Descriptors._descList]
nms.remove("TPSA")
nms.remove("MolLogP")
calc=MoleculeDescriptors.MolecularDescriptorCalculator(nms)

of=open("drug_descriptors.txt","w")
of.write("# drug_name, pubchem_id, cns, xlogp, tpsa, ")
for i in nms:
    of.write("%s, "%(i))
of.write("\n")
for i in f[1:]:
    l=i.split(",")
    #print(l)
    names.append(l[0].strip())
    cid.append(int(l[1]))
    if l[2].strip() == "TRUE":
        cns.append(True)
    else:
        cns.append(False)
    xlogp.append(float(l[3]))
    tpsa.append(float(l[4]))
    mols.append(Chem.MolFromSmiles(l[-1]))

    of.write("%30s, %10d, %8d, %8.3f, %8.3f, "%(names[-1],cid[-1],bool(cns[-1]),xlogp[-1],tpsa[-1]))
    for j in calc.CalcDescriptors(mols[-1]):
        of.write("%12.5f, "%(j))
    of.write("\n")
of.close()


