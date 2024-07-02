# <p align=center> Awesome 🎉Generative Face Video Coding </p>

<img width="1200"  alt="GFVC_framework" src="https://github.com/Berlin0610/Awesome-Generative-Face-Video-Coding/assets/80899378/9d68cc66-f86b-4e79-95e8-1658d85f1628">

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)  ![GitHub stars](https://github.com/Berlin0610/Awesome-Generative-Face-Video-Compression.svg?color=red) 

# Implementation Codes
We optimize the implemention codes of three representative GFVC works, i.e., [FOMM](https://github.com/AliaksandrSiarohin/first-order-model), [CFTE](https://github.com/Berlin0610/CFTE_DCC2022) and [FV2V](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), and further provide the unified codes regarding the encoder and decoder processes.

+ Download the `CFTE-checkpoint.pth.tar`, `FOMM-checkpoint.pth.tar`, and `FV2V-checkpoint.pth.tar` to the path `./GFVC/CFTE/checkpoint/`, `./GFVC/FOMM/checkpoint/`, and `./GFVC/FV2V/checkpoint/` respectively. The unified checkpoint dir is available at [this link](https://portland-my.sharepoint.com/:u:/g/personal/bolinchen3-c_my_cityu_edu_hk/EZ3rHarhkzhMisnJDTM7XOYBIH0lVI2jrdOK_xn_mj-tVg?e=KHfCa0).
+ The overall testing dataset is available at [this link](https://portland-my.sharepoint.com/:f:/g/personal/bolinchen3-c_my_cityu_edu_hk/En0W90hNlrZLokuzGb67lgIBMqeHSIZZHff95ZyI0-WG7g?e=1cx4ZG).
+ The specific details can be seen in the subfolder `Software`.
  
# Sample Demos
## Demo: Similar Bitrate &&  Similar Quality
[![IMAGE ALT TEXT](https://github.com/Berlin0610/Awesome-Generative-Face-Video-Compression/assets/80899378/e5bbc369-dd18-4294-bfc2-9918baa1eac3)](https://github.com/Berlin0610/Awesome-Generative-Face-Video-Compression/assets/80899378/e5bbc369-dd18-4294-bfc2-9918baa1eac3)

## Demo: Animating Virtual Faces
[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/236778089-cc2018df-9943-4b57-8514-74dfdac712df.mp4)](https://user-images.githubusercontent.com/80899378/236778089-cc2018df-9943-4b57-8514-74dfdac712df.mp4)

## Demo: Interacting with Facial Expression and Head Posture
[![IMAGE ALT TEXT](https://github.com/Berlin0610/Awesome-Generative-Face-Video-Compression/assets/80899378/db59d119-0296-49ea-8208-91a2770be04d)](https://github.com/Berlin0610/Awesome-Generative-Face-Video-Compression/assets/80899378/db59d119-0296-49ea-8208-91a2770be04d)


# Rate-distortion Performance

<img width="1100" alt="RDperformance" src="https://github.com/Berlin0610/Awesome-Generative-Face-Video-Compression/assets/80899378/76aa7da5-62da-4286-a754-ab9368e42341">


# Technical Summary

---
## <span id="Paper2024">✔2024 </span> [       «🎯Back To Top»       ](#)
- (ICIP 2024) [**GFVC_Review**] **Generative Visual Compression: A Review** Chen Bolin, Yin Shanzhi, Chen Peilin, Wang Shiqi, Ye Yan [paper](https://arxiv.org/abs/2402.02140)
- (DCC 2024) [**GFVC_Review**] **Generative Face Video Coding Techniques and Standardization Efforts: A Review** Chen Bolin, Chen Jie, Wang Shiqi, Ye Yan [paper](https://arxiv.org/pdf/2311.02649.pdf)
- (DCC 2024) [**GFVC_Translator**] **Enabling Translatability of Generative Face Video Coding: A Unified Face Feature Transcoding Framework** Yin Shanzhi, Chen Bolin, Wang Shiqi, Ye Yan

---
## <span id="Paper2023">✔2023 </span> [       «🎯Back To Top»       ](#)
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
- (MMSP 2022) [**SOM**] **Robust Ultralow Bitrate Video Conferencing with Second Order Motion Coherency** Chen Zhehao, Lu Ming, Chen Hao, Ma Zhan [paper](https://ieeexplore.ieee.org/abstract/document/9949138)

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
## <span id="Proposal2024">✔2024 </span> [       «🎯Back To Top»       ](#)

- (JVET April 2024) [**JVET-AH0016**] **JVET AHG report: Generative face video compression (AHG16)** Yan Ye, Han Boon Teo, Zhuoyi Lv, Sean McCarthy, Shiqi Wang [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=14051)
- (JVET April 2024) [**JVET-AH0053**] **AHG9/AHG16: Comments on Generative Face Video SEI** S. Deshpande (Sharp) [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=13935)
- (JVET April 2024) [**JVET-AH0054**] **AHG16: On the generative face video SEI message** H.-B. Teo, J.-Y. Thong, K. Abe, C.-S. Lim, K. Jayashree (Panasonic) [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=13936)
- (JVET April 2024) [**JVET-AH0110**] **AHG16: Scalable Representation and Layered Reconstruction for Generative Face Video Compression** B. Chen, Y. Ye, J. Chen, R.-L. Liao(Alibaba), S. Yin, S. Wang (CityU) [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=13992)
- (JVET April 2024) [**JVET-AH0113**] **AHG16: Lightweight CFTE with Multi-Resolution support** R. Zou, B. Chen, R.-L. Liao, J. Chen, Y. Ye (Alibaba)[Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=13995)
- (JVET April 2024) [**JVET-AH0114**] **AHG16: Updated Common Software Tools for Generative Face Video Compression** B. Chen, Y. Ye (Alibaba), G.Konuko, G. Valenzise (CentraleSupelec), S. Yin, S. Wang (CityU)[Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=13996)
- (JVET April 2024) [**JVET-AH0118**] **AHG9/AHG16: Showcase for picture fusion for generative face video SEI message** S. Gehlot, G. Su, P. Yin, S. McCarthy, G. J. Sullivan (Dolby)[Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=14000)
- (JVET April 2024) [**JVET-AH0127**] **AHG9/AHG16: The SEI message design for scalable representation and layered reconstruction for generative face video compression** J. Chen, B. Chen, Y. Ye, R.-L. Liao (Alibaba), S. Yin, S. Wang (CityU)[Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=14009)
- (JVET April 2024) [**JVET-AH0137**] **AHG9: Indicating timing information for generative face video (GFV) output pictures** L. Chen, O. Chubach, Y.-W. Huang, S. Lei (MediaTek)[Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=14019)
- (JVET April 2024) [**JVET-AH0138**] **AHG9/AHG16: Pupil position SEI message for Generative Face Video** A. Trioux, Y. Yao, F. Ma, F. Yang(Xidian Univ.), F. Xing, Z. Wang(Hisense)[Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=14020)
- (JVET April 2024) [**JVET-AH0148**] **AHG9/AHG16: On key point coordinates calculation for the generative face video SEI message** K. Yang (SJTU), Y.-K. Wang (Bytedance), Y. Xu, Y. Li (SJTU)[Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=14030)
- (JVET April 2024) [**JVET-AH0239**] **AHG9/AHG16: Software Implementation of Generative Face Video SEI Message** B. Chen, J. Chen, R. Zou, Y. Ye, R.-L. Liao (Alibaba), S. Wang (CityU)[Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=14138)
- (JVET April 2024) [**JVET-AH0349**] **AHG9/AHG16: On semantics fix for GFV SEI message** J. Chen, Y. Ye (Alibaba), K. Yang (SJTU), Y.-K. Wang (Bytedance), Y. Xu, Y. Li (SJTU)[Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=14249)

- (JVET January 2024) [**JVET-AG0016**] **JVET AHG report: Generative face video compression (AHG16)** Yan Ye, Han Boon Teo, Zhuoyi Lv, Sean McCarthy, Shiqi Wang [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=13789)
- (JVET January 2024) [**JVET-AG0042**] **AHG16: Proposed Common Software Tools and Testing Conditions for Generative Face Video Compression** Bolin Chen, Jie Chen, Ru-Ling Liao, Yan Ye, Shiqi Wang [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=13595)
- (JVET January 2024) [**JVET-AG0048**] **AHG16: Interoperability Study on Parameter Translator of Generative Face Video Coding** Shanzhi Yin, Bolin Chen, Jie Chen, Ru-Ling Liao, Yan Ye, Shiqi Wang [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=13602)
- (JVET January 2024) [**JVET-AG0087**] **AHG9: On the generative face video SEI message** M. M. Hannuksela, F. Cricri, H. Zhang [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=13643)
- (JVET January 2024) [**JVET-AG0088**] **AHG9: Usage of the neural-network post-filter characteristics SEI message to define the generator NN of the generative face video SEI message** M. M. Hannuksela, F. Cricri, H. Zhang [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=13644)
- (JVET January 2024) [**JVET-AG0139**] **AHG16: Depthwise separable convolution for generative face video compression** Renjie Zou, Ru-Ling Liao, Bolin Chen, Jie Chen, Yan Ye [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=13695)
- (JVET January 2024) [**JVET-AG0187**] **AHG16: Study text for common test conditions and evaluation procedures for generative face video coding (draft 1)** Sean McCarthy, Peng Yin, Bolin Chen, Yan Ye, Shiqi Wang [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=13743)
-  (JVET January 2024) [**JVET-AG0203**] **AHG9/AHG16: Common text for proposed generative face video SEI message** Jie Chen, Bolin Chen, Yan Ye, Shanzhi Yin, Shiqi Wang, Sean McCarthy, Peng Yin, Guan-Ming Su, Anustup Kuma Choudhury, Walt Husak, Gary J. Sullivan  [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=13759)
-  (JVET January 2024) [**JVET-AG2035**] **Test conditions and evaluation procedures for generative face video coding** Sean McCarthy, Bolin Chen   [Proposal](https://jvet-experts.org/doc_end_user/current_document.php?id=13920)



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
