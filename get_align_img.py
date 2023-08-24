import os
img_list = os.listdir("/home/jisijie/Code/test_imgs2_align")
f = open("test_imgs2_align.txt","w+")
for i in range(len(img_list)):
	#import pdb; pdb.set_trace()
	f.write(img_list[i] + '\n')