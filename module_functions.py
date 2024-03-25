
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
    show='once',
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
        show=show,
    )



def maxpool(
    name,
    x,
    kernel_size,
    stride,
    show='once',
    mdic=None,
):
    return create_module(
        name=name,
        type_='maxpool',
        x=x,
        kernel_size=kernel_size,
        stride=stride,
        show=show,
        mdic=mdic,
    )



def identity(x,*args,show='once',**kwargs):
    return create_module(
        name=name,
        type_='identity',
        x=x,
        show=show,
    )    



def upsample(
    name,
    x,
    image_height,
    image_width,
    mode='nearest',
    show='once',
    mdic=None,
):
    return create_module(
        name=name,
        type_='upsample',
        x=x,
        image_height=image_height,
        image_width=image_width,
        mode=mode,
        show=show,
        mdic=mdic,
    )



def linear(
    name,
    x,
    in_features=0, # should this be computed from x, as with conv2d?
    out_features=0,
    bias=True,
    device=None,
    dtype=None,
    show='once',
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
        show=show,
        mdic=mdic,
    )



def fire_module(
    name,
    x,
    squeeze_planes,
    expand1x1_planes,
    expand3x3_planes,
    show='once',
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
        show=show,
    )



def create_module(
    name,
    type_,
    x,
    mdic=None,
    show='once',
    **kwargs,
):
    assert mdic is not None
    assert len(x.size())==4
    in_channels=x.size()[1]
    #show=False
    #######################
    # store in mdic
    if not 'debug':
        cm(name,type_,x.size(),show)
    show_now=False
    if show=='always':
        show_now=True
    if name not in mdic:
        if show=='once':
            show_now=True
        if type_=='conv2d':
            if kwargs['batch_norm']:
                mdic[name+'_batch_norm']=nn.BatchNorm2d(kwargs['out_channels'])
                kwargs['bias']=False
            mdic[name]=nn.Conv2d(
                in_channels,
                kwargs['out_channels'],
                kwargs['kernel_size'],
                kwargs['stride'],
                kwargs['padding'],
                kwargs['dilation'],
                kwargs['groups'],
                kwargs['bias'],
                kwargs['padding_mode'],
                kwargs['device'],
                kwargs['dtype'],
            )
            if kwargs['activation']:
                mdic[name+'_activation']=kwargs['activation']
        elif type_=='fire':
            mdic[name]=Fire(
                kwargs['squeeze_planes'],
                kwargs['expand1x1_planes'],
                kwargs['expand3x3_planes'],
                show=show,
                mdic=mdic,
            )
        elif type_=='maxpool':
            mdic[name]=nn.MaxPool2d(
                kernel_size=kwargs['kernel_size'],
                stride=kwargs['stride'],
                ceil_mode=True,
            )
        elif type_=='upsample':
            mdic[name]=nn.Upsample(
                size=(kwargs['image_height'],kwargs['image_width']),
                mode=kwargs['mode'])
        elif type_=='identity':
            mdic[name]=nn.Identity()
        elif type_=='linear':
            mdic[name]=nn.Linear(
                in_features=ikwargs['n_features'],
                out_features=kwargs['out_features'],
                bias=kwargs['bias'],
                device=kwargs['device'],
                dtype=kwargs['dtype'],
            )
        else:
            assert False     
    #
    #######################

    #######################
    # excitation, batch norm and activation 
    y=mdic[name](x)

    if 'batch_norm' in kwargs and kwargs['batch_norm']:
        y=mdic[name+'_batch_norm'](y)

    if name+'_activation' in mdic:
        y=mdic[name+'_activation'](y)
    #
    #######################

    #######################
    # show 
    if show_now:
        if name+'_activation' in mdic:
            activation_str=str(kwargs['activation']).split('(')[0]
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
        if 'mode' in kwargs:
            mode_str=kwargs['mode']
        else:
            mode_str=''
        print(
            d2s(
                tab+type_,
                name+':',
                x_size,
                '-->',
                y_size,
                batch_norm_str,
                activation_str,
                mode_str,
                dp(y_size[1]*y_size[2]*y_size[3]/1000.,1),
                '\bk',
        ))
    #
    #######################

    return y



tensor_dictionary={}
def describe_tensor(x,name,type_='',tab='',show='once',tensor_dictionary=tensor_dictionary):
    x_size=shape_from_tensor(x)
    k=d2s(name,type,x_size)
    if show == 'never':
        return
    if k not in tensor_dictionary or show=='always':
        print(
            d2s(
                tab+type_,
                name+' --',
                x_size,
                dp(x_size[1]*x_size[2]*x_size[3]/1000.,1),
                '\bk',
        ))
        tensor_dictionary[k]=True
    
def describe_input(x,_,show,tabs=0):
    if type(show) is list:
        if _.__class__.__name__ in show:
            show='always'
        else:
            show='never'
    describe_tensor(x,d2s(tabs*'\t'+'[***',_.__class__.__name__,'INPUT'),show=show)

def describe_output(x,_,show,tabs=0):
    if type(show) is list:
        if _.__class__.__name__ in show:
            show='always'
        else:
            show='never'
    describe_tensor(x,d2s(tabs*'\t'+'. . .',_.__class__.__name__,'OUTPUT ***]'),show=show)





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
        show='once',
    ):
        super(Fire,_).__init__()
        packdict(_,locals())
    def forward(_,x):
        squeeze=    conv2d('fire input squeeze',x,_.squeeze_planes,kernel_size=1,mdic=_.mdic,show=_.show)
        expand1x1=  conv2d('fire expand1x1',squeeze,_.expand1x1_planes,kernel_size=1,mdic=_.mdic,show=_.show)
        expand3x3=  conv2d('fire expand3x3',squeeze,_.expand3x3_planes,kernel_size=3,padding=1,mdic=_.mdic,show=_.show)
        return torch.cat([expand1x1,expand3x3],1)



# batch channel hight width


#EOF
