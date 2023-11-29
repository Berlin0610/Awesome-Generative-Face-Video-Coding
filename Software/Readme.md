
# Implementation Codes
We optimize the implemention codes of three representative GFVC works, i.e., [FOMM](https://github.com/AliaksandrSiarohin/first-order-model), [CFTE](https://github.com/Berlin0610/CFTE_DCC2022) and [FV2V](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), and further provide the unified codes regarding the encoder and decoder processes.

+ Download the `CFTE-checkpoint.pth.tar`, `FOMM-checkpoint.pth.tar`, and `FV2V-checkpoint.pth.tar` to the path `./GFVC/CFTE/checkpoint/`, `./GFVC/FOMM/checkpoint/`, and `./GFVC/FV2V/checkpoint/` respectively. The unified checkpoint dir is available at [this link](https://portland-my.sharepoint.com/:u:/g/personal/bolinchen3-c_my_cityu_edu_hk/EZ3rHarhkzhMisnJDTM7XOYBIH0lVI2jrdOK_xn_mj-tVg?e=KHfCa0).
+ The overall testing dataset is available at [this link](https://portland-my.sharepoint.com/:f:/g/personal/bolinchen3-c_my_cityu_edu_hk/En0W90hNlrZLokuzGb67lgIBMqeHSIZZHff95ZyI0-WG7g?e=1cx4ZG).


In details, we provide the specific introductions about the hyper parameters and their definitions in the GFVC software tools as follows,
-	`--original_seq`: the path to the input test sequence
-	`--encoding_frames`: the number of frames to be encoded
-	`--seq_width`: "the width of encoding frames
-	`--seq_height`: the height of encoding frames
-	`--quantization_factor`: the quantization factor for the type conversion (i.e., from float-type to int-type) for the residual of feature parameter 
-	`--Iframe_QP`: the quantization parameter for encoding the base picture
-	`--Iframe_format`: the coded color format for the base picture, e.g., YUV 420 or RGB 444

## Encoding/Decoding Porcesses
The platform details can be described as follows,
-	The pretrained analysis/synthesis models and codes of the three representative GFVC algorithms are encapsulated in the `GFVC` folder. 
-	The corresponding interfaced functions regarding the encoder and decoder are defined in `CFTE_Encoder.py`, `CFTE_Decoder.py`, `FOMM_Encoder.py`, `FOMM_Decoder.py`, `FV2V_Encoder.py` and `FV2V_Decoder.py`.
-	The `arthmetic` and `vtm` folders include the packaged tools regarding the context-adaptive arithmetic coder for feature parameter encoding and the latest VVC software VTM 22.2 for base picture encoding.
-	The shell file (i.e., `RUN.sh` ) and batch execution code (i.e., `RUN.py` ) are provided to complete the encoding and decoding processes.

The usages can be provided as follows,
-	Modify the corresponding hyper parameters in `RUN.py`. The specific details have been provided in this `RUN.py`.


## Rate Calculation and Distortion Evaluation

In the folder `evaluate`, we further provide the codes to calculate the rate and distortion.
-	`multiMetric.py`: a unified code including DISTS/LPIPS/PSNR/SSIM
-	`getbits.py`: calculate the coding bits of base picture (VVC bitstream) and feature parameter (feature bitstream)


## Anchor Comparison (VTM 22.2)

VTM 22.2 is used and the test is conducted under low-delay B (LDB) configuration. And we provide the batch execution code to encode these sequences and obtain the coresponding rate-distortion performance. The codes can be seen in folder `VVC_anchor`. You can execute `RUN_Encode.py` and `RUN_Decode.py`.



# Acknowledgement
We really appreciate all authors for making their codes available to public.
- The GFVC software package includes [FOMM](https://github.com/AliaksandrSiarohin/first-order-model), [CFTE](https://github.com/Berlin0610/CFTE_DCC2022) and [FV2V](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis). 
- The testing dataset is sourced and preprocessed from [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) and [CFVQA](https://github.com/Yixuan423/Compressed-Face-Videos-Quality-Assessment).
- The quality assessment metrics include [DISTS](https://github.com/dingkeyan93/DISTS) and [LPIPS](https://github.com/richzhang/PerceptualSimilarity).

