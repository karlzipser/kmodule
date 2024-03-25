
from kmodule.module_functions import *



CA()


s=rimread(opjh('sampleimages/sunny.jpg'))[:,40:-40,:]
mdic=nn.ModuleDict()
h,w=shape(s)[0],shape(s)[1]

x=torch.zeros(1,3,h,w)
x[0,:]=torch.from_numpy(s.transpose(2,0,1))

sh(x,'x')
x1=upsample('u1',x,image_height=h//8,image_width=w//8,mdic=mdic)
sh(x1,'x1')
x0=upsample('u0',x,image_height=h//32,image_width=w//32,mdic=mdic)
sh(x0,'x0')
x2=upsample('u2',x1,image_height=h,image_width=w,mdic=mdic,mode='bilinear')
x3=upsample('u3',x0,image_height=h,image_width=w,mdic=mdic,mode='bilinear')
#sh(x2,'x2')

ctr=0
for q in [x,x0,x1,x2,x3]:
    sh(q,ctr)
    ctr+=1
sh(x+x2+x3,'x+x2+x3')

cm(r=1)




    if False:
        xs=[]
        h_final,w_final=69,131
        x_base=torch.zeros(1,16,h_final,w_final)
        widths=[512,256,128,64,32]
        for w in widths:
            cg(w)
            x=torch.from_numpy(na(rndn(bs,nin,w//2,w))).float()
            #x=conv2d('input 3x3',x,out_channels=nch*2,kernel_size=3,stride=2,padding=1,mdic=mdic)
            x=feedforward_block(x)
            x_attention=attention_block(x)
            #describe_tensor(x_attention,'x_attention')
            #describe_tensor(x,'x')
            x=torch.mul(x,x_attention)
            x=upsample('output final',x,image_height=h_final,image_width=w_final,mode='bilinear',mdic=mdic)
            x_base+=x
            print(2*'\n')
        x_base/=len(widths)
        describe_tensor(x_base,'end')


    
#EOF
