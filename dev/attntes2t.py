
from knets.nets.utils.setup import *


mdic=nn.ModuleDict()

#######################################################################
##
def conv2d(
    name,
    x,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode='zeros',
    device=None,
    dtype=None,
    mdic=mdic,
    activation=nn.ReLU(True),
    batch_norm=False,
    hide=False,
):
    assert len(x.size())==4

    in_channels=x.size()[1]

    show=False
    
    if name not in mdic:
        show=not hide
        if batch_norm:
            mdic[name+'_batch_norm']=nn.BatchNorm2d(out_channels)
            bias=False
        mdic[name]=nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        if activation:
            mdic[name+'_activation']=activation
    else:
        pass #cm(name,'already stored in module dict')



    y=mdic[name](x)


    if batch_norm:
        y=mdic[name+'_batch_norm'](y)

    if name+'_activation' in mdic:
        y=mdic[name+'_activation'](y)



    if show:
        if name+'_activation' in mdic:
            activation_str=str(activation).split('(')[0]
        else:
            activation_str=''
        if name+'_batch_norm' in mdic:
            batch_norm_str=str(mdic[name+'_batch_norm']).split('(')[0]
        else:
            batch_norm_str=''
        x_size=shape_from_tensor(x)
        y_size=shape_from_tensor(y)
        if 'input' in name.lower():# or 'output' in name.lower():
            tab=''
            #x_size=str(x_size)+'\n\t'
        else:
            tab='\t'
            x_size=''
        print(
            d2s(
                tab+name+':',
                x_size,
                '-->',
                y_size,
                batch_norm_str,
                activation_str,
                dp(y_size[1]*y_size[2]*y_size[3]/1000.,1),
                '\bk',
        ))
        
    return y
##
#######################################################################


    
w=256
bs=11

x=torch.from_numpy(na(rndn(bs,3,w,w))).float()
if False:
    a=conv2d('a_input', x,16,3,2,1,batch_norm=True)
    b=conv2d('b',       a,32,3,2,1,batch_norm=True,activation=nn.Sigmoid())
    c=conv2d('c',       b,64,3,2,1,activation=nn.LeakyReLU(0.2, inplace=True))
    d=conv2d('d_output',c,128,3,2,1,activation=None)


class Fire(nn.Module):
    def __init__(
        _,
        squeeze_planes,
        expand1x1_planes,
        expand3x3_planes,
    ):
        super(Fire,_).__init__()
        _.mdic=nn.ModuleDict()
        _.squeeze_planes=squeeze_planes
        _.expand1x1_planes=expand1x1_planes
        _.expand3x3_planes=expand3x3_planes
    def forward(_,x):
        squeeze=    conv2d('input-squeeze',x,_.squeeze_planes,kernel_size=1,mdic=_.mdic)
        expand1x1=  conv2d('expand1x1',squeeze,_.expand1x1_planes,kernel_size=1,mdic=_.mdic)
        expand3x3=  conv2d('expand3x3',squeeze,_.expand3x3_planes,kernel_size=3,padding=1,mdic=_.mdic)
        return torch.cat([expand1x1,expand3x3],1)

net=Fire(3,6,7)
net2=Fire(3,6,7)
x=torch.from_numpy(na(rndn(bs,16,w,w))).float()
x=net(x)
x=net2(x)
pass
#EOF
