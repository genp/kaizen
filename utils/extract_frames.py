#!/usr/bin/env python
from subprocess import Popen, PIPE
import os
import re
import glob

def extract_frames(vidname, dirname, min_frames=50.0):
    '''
    This function extracts individual frames from the input video and
    saves them in jpg format in the directory dirname.
    '''
    frame_fnames = []

    ffmpeg_metadata_args = ["ffmpeg",
                            "-i", vidname,
                            "-map", "0:v:0",
                            "-c", "copy",
                            "-f", "null", "-"]
    proc = Popen(ffmpeg_metadata_args, stdout=PIPE, stderr=PIPE)
    _, err = proc.communicate()
    num_frames = int(re.search("frame= (.*) fps=", str(err)).groups()[0])

    print(f'Total frames in {vidname} {num_frames}')
    print(f'Creating frames from {vidname} and saving to dir {dirname}...')
    try:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    except OSError:
        print(('Error: Creating directory of data'))
        return None

    # Extract frame_limit number of frames, which is the min of 25% of
    # the frames (to reduce near duplicates) or 50 frames
    # At least one frame will be sampled
    frame_limit = min(min_frames, max(round(num_frames*0.25), 1.0))
    try:
        fps = float(re.search("Stream #0:0(.*), (.*) fps,", str(err)).groups()[1])

    except AttributeError as e:
        # if the container information is formatted in an unexpected way,
        # estimate the fps.
        fps = 25.0

    # sampling fps is the original fps divided by the ratio of number
    # of orginal frame to disired output frames
    sample_fps = fps/ (num_frames/frame_limit)

    # Saves image of the current frame in jpg file
    print(f'sample fps: {sample_fps:.2f}')
    sample_frames_args = ["ffmpeg",
                          "-i", vidname,
                          "-vf", f"fps={sample_fps}",
                          f"{dirname}/frame_%d_fps_{sample_fps:.2f}.jpg"]
    proc = Popen(sample_frames_args, stdout=PIPE, stderr=PIPE)
    proc.wait()
    frame_fnames = glob.glob(f"{dirname}/frame_*_fps_{sample_fps:.2f}.jpg")
    # TODO: not sure if it would be faster not to wait for process to finish
    # and just return probable output fnames
    #[f"{dirname}/frame_{ind}_fps_{sample_fps:.2f}.jpg" for ind in range(frame_limit)]

    return frame_fnames
