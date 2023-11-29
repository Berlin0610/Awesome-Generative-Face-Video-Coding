
# Implementation Codes
We optimize the implemention codes of three representative GFVC works, i.e., [FOMM](https://github.com/AliaksandrSiarohin/first-order-model), [CFTE](https://github.com/Berlin0610/CFTE_DCC2022) and [FV2V](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), and further provide the unified codes regarding the encoder and decoder processes.

+ Download the `CFTE-checkpoint.pth.tar`, `FOMM-checkpoint.pth.tar`, and `FV2V-checkpoint.pth.tar` to the path `./GFVC/CFTE/checkpoint/`, `./GFVC/FOMM/checkpoint/`, and `./GFVC/FV2V/checkpoint/` respectively. The unified checkpoint dir is available at [this link](https://portland-my.sharepoint.com/:u:/g/personal/bolinchen3-c_my_cityu_edu_hk/EZ3rHarhkzhMisnJDTM7XOYBIH0lVI2jrdOK_xn_mj-tVg?e=KHfCa0).
+ The overall testing dataset is available at [this link](https://portland-my.sharepoint.com/:f:/g/personal/bolinchen3-c_my_cityu_edu_hk/En0W90hNlrZLokuzGb67lgIBMqeHSIZZHff95ZyI0-WG7g?e=1cx4ZG).


In details, we provide the specific introductions about the hyper parameters and their definitions in the GFVC software tools as follows,
-	"--original_seq": the path to the input test sequence
-	"--encoding_frames": the number of frames to be encoded
-	"--seq_width": "the width of encoding frames
-	"--seq_height": the height of encoding frames
-	"--quantization_factor": the quantization factor for the type conversion (i.e., from float-type to int-type) for the residual of feature parameter 
-	"--Iframe_QP": the quantization parameter for encoding the base picture
-	"--Iframe_format": the coded color format for the base picture, e.g., YUV 420 or RGB 444


