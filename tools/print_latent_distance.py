import torch

file1 = '/data/sss/save/modsine_xs_8_6_lr_3/epoch-100.pth'
file2 = '/data/sss/save/modsine_xs_8_6_lr_3/epoch-1900.pth'

sv_file1 = torch.load(file1)
sv_file2 = torch.load(file2)

latent1 = sv_file1['latent_list']
latent2 = sv_file2['latent_list']

small1 = latent1[:10]
first_vec = small1[0]
metric = torch.nn.CosineSimilarity(dim=0)
for vec in small1[1:]:
    print(metric(first_vec, vec))

print('============')

#small2 = list(latent2.values())[:10]
small2 = latent2[:10]
first_vec = small2[0]
metric = torch.nn.CosineSimilarity(dim=0)
for vec in small2[1:]:
    print(metric(first_vec, vec))
    

