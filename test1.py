
from kmodule.module_functions import *






class Fire_with_Attention(nn.Module):
    def __init__(
        _,
        nch,
        mdic,
    ):
        super(Fire_with_Attention,_).__init__()
        _.mdic=mdic
        _.nch=nch
        
    def forward(_,x):
        nch=_.nch
        
        x=fire_module('input module A',x,nch,nch,nch,mdic=_.mdic)
        x0=fire_module('module B',x,nch,nch,nch,mdic=_.mdic)
        h=image_height=x0.size()[-2]
        w=image_width=x0.size()[-1]
        x=maxpool('attention input',x0,kernel_size=7,stride=3,mdic=_.mdic)
        x=conv2d('7x7',x,out_channels=nch*2,kernel_size=7,stride=3,padding=3,mdic=_.mdic)
        x=fire_module('attention module',x,nch,nch,nch,mdic=_.mdic)

        x=upsample('u1',x,image_height=h,image_width=w,mdic=_.mdic)

        x1=conv2d('u1 3x3',x,out_channels=nch*2,kernel_size=3,stride=1,padding=1,mdic=_.mdic,activation=nn.Sigmoid())

        x0=upsample('u1',x0,image_height=h,image_width=w,mdic=_.mdic)

        x=torch.mul(x0,x1)

        return x


if False:
    a=torch.rand(2,3,5,5)
    b=a.view(a.size()[0],-1).size()

if __name__=='__main__':

    print(10*'\n')

    bs=1
    nin=3
    nch=8 
    mdic=nn.ModuleDict()
    net=Fire_with_Attention(nch,mdic)

    for w in [512,256,128,64,32]:
        x=torch.from_numpy(na(rndn(bs,nin,w//2,w))).float()
        x=conv2d('input 3x3',x,out_channels=nch*2,kernel_size=3,stride=2,padding=1,mdic=mdic)
        print(x.size())
        x=net(x)
        cg(x.size())

    print(4*'\n')
    
#EOF
