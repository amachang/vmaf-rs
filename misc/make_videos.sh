#!/bin/bash

set -eux

function make_y4m {
    path="$1"
    n_frames="$2"
    width="$3"
    height="$4"
    if [ ! -f "$path" ]; then
        gst-launch-1.0 -e \
            videotestsrc num-buffers="$n_frames" ! \
            'video/x-raw,format=(string)I420,width='"$width"',height='"$height"',framerate=(fraction)30/1' ! \
            y4menc ! \
            filesink location="$path"
    fi
}

function encode_h265() {
    original_path="$1"
    path="$2"
    crf="$3"
    if [ ! -f "$path" ]; then
        gst-launch-1.0 -e \
            qtmux name=mux ! \
            filesink location="$path" \
            \
            filesrc location="$original_path" ! \
            y4mdec ! \
            x265enc option-string=crf="$crf" ! \
            h265parse ! \
            mux.video_0
    fi
}

function encode_decode_y4m {
    original_path="$1"
    path="$2"
    crf="$3"
    if [ ! -f "$path" ]; then
        # y4menc breaks y4m format when there's zero padding at the end of input_buffer.
        # And avdec_h265 padded zero at the end of input_buffer,
        # so we couldn't use avdec_h264, and use libde265dec here.
        # Seems to be a bug of y4menc.
        # Related issues: https://gitlab.freedesktop.org/gstreamer/gstreamer/-/issues/2765

        gst-launch-1.0 -e \
            filesrc location="$original_path" ! \
            y4mdec ! \
            x265enc option-string=crf="$crf" ! \
            libde265dec ! \
            y4menc ! \
            filesink location="$path"
    fi
}

make_y4m videos/original.y4m 300 320 240
encode_h265 videos/original.y4m videos/low_quality.mp4 51
encode_h265 videos/original.y4m videos/high_quality.mp4 10
encode_decode_y4m videos/original.y4m videos/low_quality.y4m 51
encode_decode_y4m videos/original.y4m videos/high_quality.y4m 10

make_y4m videos/short_original.y4m 2 160 120
encode_decode_y4m videos/short_original.y4m videos/short_low_quality.y4m 51
encode_decode_y4m videos/short_original.y4m videos/short_high_quality.y4m 10

# To have files with a meaningless name, for example codes
if [ ! -f videos/foo.y4m ]; then
    cp videos/short_original.y4m videos/foo.y4m
fi
if [ ! -f videos/bar.y4m ]; then
    cp videos/short_high_quality.y4m videos/bar.y4m
fi

