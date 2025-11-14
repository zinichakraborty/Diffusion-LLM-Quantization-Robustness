import torch
import matplotlib.pyplot as plt
import statistics


dt = torch.load("/storage/ice1/0/7/agupta965/research2/hawq/sft/checkpoints/final_sensitivity_analysis.pt")
dlt = dt['hessian_info']
dat = []
for x,y in dlt.items():
    bid = ".".join(x.split(".")[-4:-1])
    print(f"layername | {bid} || sensitivity | {y['sensitivity']}")
    dat.append((bid,y['sensitivity']))

sens = [x[1] for x in dat]
print(f"max {max(sens)} | min {min(sens)} | avg {sum(sens) / len(sens)} | median {statistics.median(sens)}")
print(f"N={len(sens)}")

dat = dat[:25]
print(dat)
fig, ax = plt.subplots(figsize=(12, 8))
plt.bar([x[0] for x in dat ], [x[1] for x in dat])
plt.suptitle('Estimated 25 Most Sensitive Layers in CoDA 2B during Finetuning on GSM')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.subplots_adjust(bottom=0.2, left=0.2)  # adjust as needed, e.g. 0.3
plt.xlabel("Block, Layer of Weight")
plt.ylabel("Estimated Sensitivity by Hessian Eigenvalues")
plt.savefig('foo.png',dpi=400)

print("Figure saved")