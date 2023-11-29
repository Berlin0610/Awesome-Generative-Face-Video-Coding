#!/bin/sh


./bin/EncoderAppStatic_VTM22_2 -c ./cfg/encoder_lowdelay_vtm.cfg -c ./cfg/per-sequence/inputyuv420.cfg -q $1 -f $2 -wdt $3 -hgt $4 -i $5 -o $6/$7_qp$1.yuv -b $6/$7_qp$1.bin >>$6/$7_qp$1.log 

