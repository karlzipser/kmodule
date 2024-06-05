
from kmodule.module_functions import *


class _Fire_with_Attention(nn.Module):
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





class Feedforward_Block(nn.Module):
    def __init__(
        _,
        nch,
        mdic=nn.ModuleDict()
    ):
        super(Feedforward_Block,_).__init__()
        packdict(_,locals())
    def forward(_,x):
        x=fire_module('feedforward A',x,_.nch,_.nch,_.nch,mdic=_.mdic,hide=False)
        #x=fire_module('feedforward B',x,_.nch,_.nch,_.nch,mdic=_.mdic,hide=False)
        return x



class Attention_Block(nn.Module):
    def __init__(
        _,
        nch,
        mdic,
    ):
        super(Feedforward_Block,_).__init__()
        packdict(_,locals())
    def forward(_,x_in):
        h=image_height=x.size()[-2]
        w=image_width=x.size()[-1]
        x=maxpool('attention input',x_in,kernel_size=7,stride=3,mdic=_.mdic)
        x=conv2d('attention 7x7',x,out_channels=nch*2,kernel_size=7,stride=3,padding=3,mdic=_.mdic)
        x=upsample('attention u1',x,image_height=h,image_width=w,mdic=_.mdic)
        x=conv2d('attention 3x3',x,out_channels=nch*2,kernel_size=3,stride=1,padding=1,mdic=_.mdic,activation=nn.Sigmoid())
        x=torch.mul(x,x_in)




if __name__=='__main__':

    print(10*'\n')

    bs=1
    nin=3
    nch=8 
    net=Feedforward_Block(nch)
    mdic=nn.ModuleDict()
    for w in [512]:#,256,128,64,32]:
        x=torch.from_numpy(na(rndn(bs,nin,w//2,w))).float()
        x=conv2d('input 3x3',x,out_channels=nch*2,kernel_size=3,stride=2,padding=1,mdic=mdic)
        #print(x.size())
        x=net(x)
        cg(x.size())

    print(4*'\n')



if False:

    class B(nn.Module):
        def __init__(_,a=1,b=2):
            super(B,_).__init__()
            packdict(_,locals())
        def forward(_,x):
            print(_.a,_.b)
            return x
    b=B();b.forward(0)



    class A(nn.Module):
        def __init__(
            _,
            squeeze_planes,
            expand1x1_planes,
            expand3x3_planes,
            mdic=nn.ModuleDict(),
            hide=False,
        ):
            super(A,_).__init__()
            packdict(_,locals())

        def forward(_,x):
            squeeze=    conv2d('squeeze',x,_.squeeze_planes,kernel_size=1,mdic=_.mdic,hide=_.hide)
            expand1x1=  conv2d('expand1x1',squeeze,_.expand1x1_planes,kernel_size=1,mdic=_.mdic,hide=_.hide)
            expand3x3=  conv2d('expand3x3',squeeze,_.expand3x3_planes,kernel_size=3,padding=1,mdic=_.mdic,hide=_.hide)
            return torch.cat([expand1x1,expand3x3],1)
    a=A(5,5,5,hide=True)
    bs=1
    nin=3
    h=128
    w=256
    x=torch.from_numpy(na(rndn(bs,nin,h,w))).float()
    print(x.size())
    x=a.forward(x)
    print(x.size())
#EOF
