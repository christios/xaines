from __future__ import unicode_literals
import os
import youtube_dl
import argparse


class MyLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print(msg)


def my_hook(d):
    print(d)
    if d['status'] == 'finished':
        print('Done downloading, now converting ...')


def adjust_ydl_options(last_dir, topic_dir):
    ydl_opts = {
        'download_archive': f'{last_dir}/downloaded_videos.txt',
        'logger': MyLogger(),
        'progress_hooks': [my_hook],
        'ignoreerrors': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'format': '18',
        'noplaylist': True,
        'outtmpl': f'{topic_dir}/%(id)s.%(ext)s',
    }
    return ydl_opts


def read_video_list(infile):
    out = []
    with open(infile, 'r') as fs:
        for f in fs:
            out.append('www.youtube.com/watch?v='+f.replace('\n', ''))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default='/hd2/data/cennet/xaines/data', type=str,
                        help="Absolute path of the directory where you want to store the videos. The videos will be stored in a folder with the same name as <video_ids> and the videos inside will be stored with the same folder hierarchy as the IDs are found in <video_ids>.")
    parser.add_argument("--video_ids", default='/home/cayralat/youtube_videos/video_ids', type=str,
                        help="Absolute path of the directory which contains the video IDs")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    root = args.root_dir
    ids_dir = args.video_ids
    last_dir = ''
    for directory in os.walk(ids_dir):
        output_dir = os.path.join(
            root, last_dir, os.path.split(directory[0])[-1])
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        # If the current directoy contains directories
        if directory[1]:
            last_dir = output_dir
            continue
        for file in os.listdir(directory[0]):
            topic_dir = os.path.join(output_dir, file)
            if not os.path.isdir(topic_dir):
                os.mkdir(topic_dir)
            ydl_opts = adjust_ydl_options(last_dir, topic_dir)
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                video_links = read_video_list(os.path.join(directory[0], file))
                ydl.download(video_links)
