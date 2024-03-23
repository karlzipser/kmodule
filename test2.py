
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




    
#EOF
