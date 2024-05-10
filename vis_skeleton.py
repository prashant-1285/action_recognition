
import glob
from pyskl.smp import *
from pyskl.utils.visualize import Vis3DPose, Vis2DPose
from mmcv import load, dump
annotations=load('/home/prashant/Documents/legion/personal_projects/action/pyskl/shoot.pkl')
print("anotations",annotations)
index = 0
anno = annotations[index]
vid = Vis2DPose(anno, thre=0.2, out_shape=(540, 960), layout='coco', fps=12, video=None)
#vid.ipython_display()