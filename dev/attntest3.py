
from .create_module import *


if __name__=='__main__':
  
    w=256
    bs=11
    mdic=nn.ModuleDict()
    #x=torch.from_numpy(na(rndn(bs,3,w,w))).float()
    #a=conv2d('a_input', x,16,3,2,1,batch_norm=True,mdic=mdic)
    #b=conv2d('b',       a,32,3,2,1,batch_norm=True,activation=nn.Sigmoid(),mdic=mdic)
    #c=conv2d('c',       b,64,3,2,1,activation=nn.LeakyReLU(0.2,inplace=True),mdic=mdic)
    #d=conv2d('d_output',c,128,3,2,1,activation=None,mdic=mdic)
    #print('-----------')

    x=torch.from_numpy(na(rndn(bs,3,w,w))).float()
    x=conv2d('a_input',x,16,3,2,1,batch_norm=False,mdic=mdic)
    x=fire_module(
        'fire module A',
        x,
        8,
        8,
        8,
        mdic=mdic,
    )
    x=fire_module(
        'fire module A',
        x,
        8,
        8,
        8,
        mdic=mdic,
    )
    cg(shape_from_tensor(x))
#EOF
