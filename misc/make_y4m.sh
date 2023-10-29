#!/bin/bash

set -eux

mkdir tests || true
rm tests/original.y4m || true
rm tests/low_quality.y4m || true
rm tests/high_quality.y4m || true

gst-launch-1.0 -e -v \
    videotestsrc num-buffers=300 ! \
    'video/x-raw,format=(string)I420,width=320,height=240,framerate=(fraction)30/1' ! \
    y4menc ! \
    filesink location=tests/original.y4m

# y4menc breaks y4m format when there's zero padding at the end of input_buffer.
# And avdec_h265 padded zero at the end of input_buffer,
# so we couldn't use avdec_h264, and use libde265dec here.
# Seems to be a bug of y4menc.
# Related issues: https://gitlab.freedesktop.org/gstreamer/gstreamer/-/issues/2765

gst-launch-1.0 -e -v \
    filesrc location=tests/original.y4m ! \
    y4mdec ! \
    x265enc option-string=crf=51 ! \
    libde265dec ! \
    y4menc ! \
    filesink location=tests/low_quality.y4m

gst-launch-1.0 -e -v \
    filesrc location=tests/original.y4m ! \
    y4mdec ! \
    x265enc option-string=crf=10 ! \
    libde265dec ! \
    y4menc ! \
    filesink location=tests/high_quality.y4m

