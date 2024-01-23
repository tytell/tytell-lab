#!/bin/bash

# run with 
# $ nohup ./convert_cines.sh &> convert_cines.out &
# then you can log out or run
# $ tail -f convert_cines.out
# to display the output

set -e

export PATH=/volume1/@appstore/ffmpeg/bin:$PATH

ffmpeg -n -i "./FlowTankCalib01.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib01_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib02.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib02_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib03.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib03_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib04.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib04_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib05.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib05_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib06.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib06_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib07.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib07_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib08.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib08_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib09.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib09_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib10.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib10_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib11.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib11_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib12.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib12_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib13.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib13_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib14.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib14_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib15.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib15_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib16.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib16_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib17.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib17_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib18.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib18_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib19.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib19_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib20.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib20_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib21.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib21_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib22.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib22_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib23.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib23_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib24.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib24_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib25.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib25_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib26.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib26_%03d.tiff"
ffmpeg -n -i "./FlowTankCalib27.cine" -ss  00:00:00 -vframes 10 "./FlowTankCalib27_%03d.tiff"