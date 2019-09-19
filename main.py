import train
import dataset
import option
import torch
import models
import time
torch.backends.cudnn.benchmark = True

opt = option.Option()
celeba_dataset = dataset.CelebaDataset_h5py(opt)
dataloader = torch.utils.data.DataLoader(celeba_dataset,opt.batchsize,shuffle=True,num_workers=16,drop_last=True,pin_memory=True)
gan = train.COCOGAN(opt)


for e in range(0,1):
    start = time.time()
    for real_macro_list in dataloader:
        gan.latent_ebdy_generator.get_latent()
        for pos,real_macro in enumerate(real_macro_list):
            gan.train_serial(real_macro,pos)
    print(time.time()-start)
    print(e)
    gan.generate_serial()
    gan.show_loss()
#     gan.save_network(e)         