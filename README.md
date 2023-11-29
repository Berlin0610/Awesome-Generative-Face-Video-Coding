# <p align=center> Awesome 🎉Generative Face Video Coding </p>

<img width="1200"  alt="GFVC_framework" src="https://github.com/Berlin0610/Awesome-Generative-Face-Video-Coding/assets/80899378/9d68cc66-f86b-4e79-95e8-1658d85f1628">

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)  ![GitHub stars](https://github.com/Berlin0610/Awesome-Generative-Face-Video-Compression.svg?color=red) 

# Sample Demos
## Demo: Similar Bitrate &&  Similar Quality
[![IMAGE ALT TEXT](https://github.com/Berlin0610/Awesome-Generative-Face-Video-Compression/assets/80899378/e5bbc369-dd18-4294-bfc2-9918baa1eac3)](https://github.com/Berlin0610/Awesome-Generative-Face-Video-Compression/assets/80899378/e5bbc369-dd18-4294-bfc2-9918baa1eac3)

## Demo: Animating Virtual Faces
[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/236778089-cc2018df-9943-4b57-8514-74dfdac712df.mp4)](https://user-images.githubusercontent.com/80899378/236778089-cc2018df-9943-4b57-8514-74dfdac712df.mp4)

## Demo: Interacting with Facial Expression and Head Posture
[![IMAGE ALT TEXT](https://github.com/Berlin0610/Awesome-Generative-Face-Video-Compression/assets/80899378/db59d119-0296-49ea-8208-91a2770be04d)](https://github.com/Berlin0610/Awesome-Generative-Face-Video-Compression/assets/80899378/db59d119-0296-49ea-8208-91a2770be04d)



# Rate-distortion Performance

<img width="1100" alt="RDperformance" src="https://github.com/Berlin0610/Awesome-Generative-Face-Video-Compression/assets/80899378/76aa7da5-62da-4286-a754-ab9368e42341">


# Implementation Codes
We optimize the implemention codes of three representative GFVC works, i.e., [FOMM](https://github.com/AliaksandrSiarohin/first-order-model), [CFTE](https://github.com/Berlin0610/CFTE_DCC2022) and [FV2V](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), and further provide the unified codes regarding the encoder and decoder processes.

+ Download the `CFTE-checkpoint.pth.tar`, `FOMM-checkpoint.pth.tar`, and `FV2V-checkpoint.pth.tar` to the path `./GFVC/CFTE/checkpoint/`, `./GFVC/FOMM/checkpoint/`, and `./GFVC/FV2V/checkpoint/` respectively. The unified checkpoint dir is available at [this link](https://portland-my.sharepoint.com/:u:/g/personal/bolinchen3-c_my_cityu_edu_hk/EZ3rHarhkzhMisnJDTM7XOYBIH0lVI2jrdOK_xn_mj-tVg?e=KHfCa0).

# Technical Summary

---
## <span id="Paper2023">✔2023 </span> [       «🎯Back To Top»       ](#)
- (arXiv 2023) [**GFVC_Review**] **Generative Face Video Coding Techniques and Standardization Efforts: A Review** Chen Bolin, Chen Jie, Wang Shiqi, Ye Yan [paper](https://arxiv.org/pdf/2311.02649.pdf)
- (ICIP 2023) [**RDAC**] **Predictive Coding for Animation-Based Video Compression** Goluck Konuko, Stéphane Lathuilière, Giuseppe Valenzise [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10222205)
- (arXiv 2023) [**IFVC**] **Interactive Face Video Coding: A Generative Compression Framework** Chen Bolin, Wang Zhao, Li Binzhe, Wang Shurun, Wang Shiqi, Ye Yan [paper](https://arxiv.org/pdf/2302.09919.pdf)
- (TCSVT 2023) [**CTTR**] **Compact Temporal Trajectory Representation for Talking Face Video Compression** Chen Bolin, Wang Zhao, Li Binzhe, Wang Shiqi, Ye Yan [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10109861)
  
---
## <span id="Paper2022">✔2022 </span> [       «🎯Back To Top»       ](#)
- (ICME 2022) [**Bi-Net**] **Generative Compression for Face Video: A Hybrid Scheme** Anni Tang, Yan Huang, Jun Ling, Zhiyu Zhang, Yiwei Zhang, Rong Xie, Li Song [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9859867)
- (BMVC 2022) [**CVC_STR**] **Compressing Video Calls using Synthetic Talking Heads** Madhav Agarwal, Anchit Gupta, Rudrabha Mukhopadhyay, Vinay P. Namboodiri, C V Jawahar [paper](https://arxiv.org/pdf/2210.03692.pdf)
- (ICIP 2022) [**HDAC**] **A Hybrid Deep Animation Codec for Low-bitrate Video Conferencing** Goluck Konuko, Stéphane Lathuilière, Giuseppe Valenzise [paper](https://arxiv.org/pdf/2207.13530.pdf)
- (ICIP 2022) [**DMRGP**] **Dynamic Multi-Reference Generative Prediction for Face Video Compression** Wang Zhao, Chen Bolin, Ye Yan, Wang Shiqi [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9897729)
- (CVPRW 2022) [**MAX-RS**] **Neural Face Video Compression Using Multiple Views** Anna Volokitin, Stefan Brugger, Ali Benlalah, Sebastian Martin, Brian Amberg, Michael Tschannen [paper](https://openaccess.thecvf.com/content/CVPR2022W/CLIC/papers/Volokitin_Neural_Face_Video_Compression_Using_Multiple_Views_CVPRW_2022_paper.pdf)
- (DCC 2022) [**C3DFD**] **Towards Ultra Low Bit-Rate Digital Human Character Communication via Compact 3D Face Descriptors** Li Binzhe, Chen Bolin, Wang Zhao, Wang Shiqi, Ye Yan [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9810765)
- (DCC 2022) [**CFTE**] **Beyond Keypoint Coding: Temporal Evolution Inference with Compact Feature Representation for Talking Face Video Compression** Chen Bolin, Wang Zhao, Li Binzhe, Lin Rongqun, Wang Shiqi, Ye Yan [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9810732)
- (DCC 2022) [**SNRVC**] **Semantic Neural Rendering-based Video Coding: Towards Ultra-Low Bitrate Video Conferencing** Hu Yujie, Xu Youmin, Chang Jianhui, Zhang Jian [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9810784)

---
## <span id="Paper2021">✔2021 </span> [       «🎯Back To Top»       ](#)
- (CVPRW 2021) [**Mob M-SPADE**] **Low Bandwidth Video-Chat Compression Using Deep Generative Models** Maxime Oquab, Pierre Stock, Daniel Haziza, Tao Xu, Peizhao Zhang, Onur Celebi, Yana Hasson, Patrick Labatut, Bobo Bose-Kolanu, Thibault Peyronel, Camille Couprie [paper](https://openaccess.thecvf.com/content/CVPR2021W/MAI/papers/Oquab_Low_Bandwidth_Video-Chat_Compression_Using_Deep_Generative_Models_CVPRW_2021_paper.pdf)
- (CVPR 2021) [**Face_vid2vid**] **One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing** Wang Ting-Chun, Mallya Arun, Liu Ming-Yu [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_One-Shot_Free-View_Neural_Talking-Head_Synthesis_for_Video_Conferencing_CVPR_2021_paper.pdf)
- (ICMEW 2021) [**VSBNet**] **A Generative Compression Framework For Low Bandwidth Video Conference** Feng Dahu, Huang Yan, Zhang Yiwei, Ling Jun, Tang Anni, Song Li [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9455985)
- (ICASSP 2021) [**DAC**] **Ultra-Low Bitrate Video Conferencing Using Deep Image Animation** Goluck Konuko, Giuseppe Valenzise, Stéphane Lathuilière [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9414731)

---
## <span id="Paper2019">✔2019 </span> [       «🎯Back To Top»       ](#)
- (NeurIPS 2021) [**FOMM**] **First Order Motion Model for Image Animation** Aliaksandr Siarohin, Stéphane Lathuilière, Sergey Tulyakov, Elisa Ricci, Nicu Sebe [paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/31c0b36aef265d9221af80872ceb62f9-Paper.pdf)


# Standardization Efforts

---
## <span id="Proposal2023">✔2023 </span> [       «🎯Back To Top»       ](#)
- (MPEG October 2023) [**m64987**] **On VVC-assisted Ultra-low Rate Generative Face Video Coding** Yan Ye, Sean McCarthy, Han Boon Teo, Zhuoyi Lv, Shiqi Wang, Kai Zhang, Marta Karczewicz, Iole Moccagatta [Proposal](https://dms.mpeg.expert/doc_end_user/current_document.php?id=89608&id_meeting=0)
- (JVET October 2023) [**JVET-AF0234**] **AHG9: Common text for proposed generative face video SEI message** Bolin Chen, Jie Chen, Yan Ye, Shiqi Wang, Sean McCarthy, Peng Yin, Guan-Ming Su, Anustup Kuma Choudhury, Walt Husak, Gary J. Sullivan [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=13497)
- (JVET October 2023) [**JVET-AF0146**] **AHG9: On Face Motion Information for Generative Face Video** Han Boon Teo, Jing Yuan Thong, Karlekar Jayashree, Chong Soon Lim, Kiyofumi Abe [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=13404)
- (JVET October 2023) [**JVET-AF0048**] **A Study on Decoder Interoperability of Generative Face Video Compression** Bolin Chen, Shanzhi Yin, Jie Chen, Yan Ye, Shiqi Wang [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=13296)
- (JVET July 2023) [**JVET-AE0280**] **AHG9: Common text for proposed generative face video SEI message** Bolin Chen, Jie Chen, Yan Ye, Shiqi Wang, Sean McCarthy, Peng Yin, Guan-Ming Su, Anustup Kuma Choudhury, Walt Husak [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=13243)
- (JVET July 2023) [**JVET-AE0088**] **AHG9: A study on Generative Face Video SEI Message** Han Boon Teo, Jing Yuan Thong, Karlekar Jayashree, Chong Soon Lim, Kiyofumi Abe [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=13036)
- (JVET July 2023) [**JVET-AE0083**] **AHG9: Common SEI Message of Generative Face Video** Bolin Chen, Jie Chen, Yan Ye, Shiqi Wang [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=13031)
- (JVET July 2023) [**JVET-AE0080**] **AHG9: Generative Face Video SEI message** Sean McCarthy, Peng Yin, Guan-Ming Su, Anustup Kuma Choudhury, Walt Husak [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=13028)
- (JVET April 2023) [**JVET-AD0051**] **AHG9: Common SEI Message of Generative Face Video** Bolin Chen, Jie Chen, Yan Ye, Shiqi Wang [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=12598)
- (JVET January 2023) [**JVET-AC0088**] **AHG9: Generative Face Video SEI Message** Bolin Chen, Jie Chen, Shurun Wang, Yan Ye, Shiqi Wang [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=12290)

# Additional Notes

- The DCC 2021 keynote video regarding [Generative Face Video Compression: Promises and Challenges](https://www.youtube.com/watch?v=7en3YYT1QfU) was presented by Dr. Yan Ye.
- The presentation slides from Dr. Yan Ye can be found in [DIS talk: Face video compression with generative networks](https://github.com/Berlin0610/DIS_talk-Yan_Ye-Face_video_compression_with_generative_networks/blob/main/DIS%20presentation%20-%20May%203%2C%202023.pdf).


# Acknowledgement
We really appreciate all authors for making their codes available to public.
- The GFVC software package includes [FOMM](https://github.com/AliaksandrSiarohin/first-order-model), [CFTE](https://github.com/Berlin0610/CFTE_DCC2022) and [FV2V](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis). 
- The testing dataset is sourced and preprocessed from [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) and [CFVQA](https://github.com/Yixuan423/Compressed-Face-Videos-Quality-Assessment).
- The quality assessment metrics include [DISTS](https://github.com/dingkeyan93/DISTS) and [LPIPS](https://github.com/richzhang/PerceptualSimilarity).
