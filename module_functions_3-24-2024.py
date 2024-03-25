
from knets.nets.utils.setup import *


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
    mdic=None,
    activation=nn.ReLU(True),
    batch_norm=False,
    hide=False,
):
    return create_module(
        name=name,
        type_='conv2d',
        x=x,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode,
        device=device,
        dtype=dtype,
        mdic=mdic,
        activation=activation,
        batch_norm=batch_norm,
        hide=hide,
    )



def maxpool(
    name,
    x,
    kernel_size,
    stride,
    hide=False,
    mdic=None,
):
    return create_module(
        name=name,
        type_='maxpool',
        x=x,
        kernel_size=kernel_size,
        stride=stride,
        hide=hide,
        mdic=mdic,
    )



def identity(x,*args,**kwargs):
    return create_module(
        name=name,
        type_='identity',
        x=x,
        hide=False,
    )    



def upsample(
    name,
    x,
    image_height,
    image_width,
    mode='nearest',
    hide=False,
    mdic=None,
):
    return create_module(
        name=name,
        type_='upsample',
        x=x,
        image_height=image_height,
        image_width=image_width,
        mode=mode,
        hide=hide,
        mdic=mdic,
    )



def linear(
    name,
    x,
    in_features=0,
    out_features=0,
    bias=True,
    device=None,
    dtype=None,
    hide=False,
    mdic=None,
):
    return create_module(
        name=name,
        type_='linear',
        x=x,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        device=device,
        dtype=dtype,
        hide=hide,
        mdic=mdic,
    )



def fire_module(
    name,
    x,
    squeeze_planes,
    expand1x1_planes,
    expand3x3_planes,
    hide=False,
    mdic=None,
):
    return create_module(
        name=name,
        type_='fire',
        x=x,
        squeeze_planes=squeeze_planes,
        expand1x1_planes=expand1x1_planes,
        expand3x3_planes=expand3x3_planes,
        mdic=mdic,
        hide=hide,
    )



def create_module(
    name,
    type_,
    x,
    out_channels=0,
    kernel_size=0,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode='zeros',
    device=None,
    dtype=None,
    mdic=None,
    activation=nn.ReLU(True),
    batch_norm=False,
    hide=False,
    squeeze_planes=0,
    expand1x1_planes=0,
    expand3x3_planes=0,
    image_height=0,
    image_width=0,
    mode='',
    in_features=0,
    out_features=0,
):
    assert len(x.size())==4
    in_channels=x.size()[1]
    show=False
    #######################
    # store in mdic
    if name not in mdic:
        show=not hide
        if type_=='conv2d':
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
        elif type_=='fire':
            mdic[name]=Fire(
                squeeze_planes,
                expand1x1_planes,
                expand3x3_planes,
                hide=hide,
                mdic=mdic,
            )
        elif type_=='maxpool':
            mdic[name]=nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                ceil_mode=True,
            )
        elif type_=='upsample':
            mdic[name]=nn.Upsample(size=(image_height,image_width),mode=mode)
        elif type_=='identity':
            mdic[name]=nn.Identity()
        elif type_=='linear':
            mdic[name]=nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                device=device,
                dtype=dtype,
            )
        else:
            assert False     
    #
    #######################

    #######################
    # excitation, batch norm and activation 
    y=mdic[name](x)

    if batch_norm:
        y=mdic[name+'_batch_norm'](y)

    if name+'_activation' in mdic:
        y=mdic[name+'_activation'](y)
    #
    #######################

    #######################
    # show 
    if True:#show:
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
        if 'input' in name.lower() or 'output' in name.lower():
            tab=''
            #x_size=str(x_size)+'\n\t'
        else:
            tab='\t'
            x_size=''
        print(
            d2s(
                tab+type_,
                name+':',
                x_size,
                '-->',
                y_size,
                batch_norm_str,
                activation_str,
                mode,
                dp(y_size[1]*y_size[2]*y_size[3]/1000.,1),
                '\bk',
        ))
    #
    #######################

    return y



tensor_dictionary={}
def describe_tensor(x,name,type_='',tab='',tensor_dictionary=tensor_dictionary):
    x_size=shape_from_tensor(x)
    k=d2s(name,type,x_size)
    if True:#k not in tensor_dictionary:
        print(
            d2s(
                tab+type_,
                name+' --',
                x_size,
                dp(x_size[1]*x_size[2]*x_size[3]/1000.,1),
                '\bk',
        ))
        tensor_dictionary[k]=True
    





def packdict(_,locals_):
    _._initial_keys=kys(_.__dict__)
    for k in locals_:
        if k[0]!='_':
            _.__dict__[k]=locals_[k]



class Fire(nn.Module):
    def __init__(
        _,
        squeeze_planes,
        expand1x1_planes,
        expand3x3_planes,
        mdic,
        hide=False,
    ):
        super(Fire,_).__init__()
        packdict(_,locals())
    def forward(_,x):
        
        squeeze=    conv2d('fire input squeeze',x,_.squeeze_planes,kernel_size=1,mdic=_.mdic,hide=False)
        expand1x1=  conv2d('fire expand1x1',squeeze,_.expand1x1_planes,kernel_size=1,mdic=_.mdic,hide=False)
        expand3x3=  conv2d('fire expand3x3',squeeze,_.expand3x3_planes,kernel_size=3,padding=1,mdic=_.mdic,hide=False)
        return torch.cat([expand1x1,expand3x3],1)



# batch channel hight width


#EOF
