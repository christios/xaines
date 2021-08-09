from __future__ import annotations
from collections import abc, Counter
from tqdm import tqdm
import pickle
import os
from typing import TextIO, Dict, Generator, Tuple

from helpers import Video, Caption, Word
from utils import *


BODY_PARTS = ['arm', 'eye', 'eyebrow', 'belly', 'leg', 'breast', 'thumb', 'elbow',
              'finger', 'foot', 'ankle', 'buttocks', 'hair', 'neck', 'face',
              'hand', 'wrist', 'hip', 'chin', 'knee', 'head', 'lip', 'mouth', 'nose',
              'nostril', 'thigh', 'ear', 'bottom', 'bum', 'back', 'underarm', 'forearm',
              'leg', 'shoulder', 'forehead', 'waist', 'calf', 'cheek',
              'eyelash', 'lash', 'tooth', 'toe', 'tongue', 'muscle', 'lung', 'spine', 'stomach',
              'chest', 'abdominal', 'ab', 'hamstring', 'quadricep', 'quad', 'glute', 'bicep',
              'tricep', 'forearm', 'trap']
BODY_PARTS_PLURAL = ['feet', 'calves', 'teeth'] + [bp + 's' for bp in BODY_PARTS]
BODY_PARTS += BODY_PARTS_PLURAL
BODY_PARTS = {bp: None for bp in BODY_PARTS}


class SubtitleReader:
    def __init__(self,
                 vtt_folder: TextIO,
                 save_path: TextIO = '') -> None:
        self.vtt_folder = vtt_folder
        self.videos, self.id_to_vid = self.read_videos(vtt_folder)
        self.videos.assign_features(save_path)

        if save_path:
            self.save(save_path)


    def read_videos(self, vtt_folder: TextIO):
        def set_leaf(tree, branches, leaf, bar):
            """Example:
            set_leaf(t, ['b1','b2','b3'], 'new_leaf') # {'b1': {'b2': {'b3': 'new_leaf'}}}
            """
            if len(branches) == 1:
                tree[branches[0]] = leaf
                return
            if not tree.get(branches[0]):
                tree[branches[0]] = {}
            set_leaf(tree[branches[0]], branches[1:], leaf, bar)

        startpath = vtt_folder
        tree: Dict = {}
        id_to_vid: Dict[str, Video] = {}
        bar = tqdm(total=sum(len(x[2]) for x in list(os.walk(startpath))))
        for root, dirs, files in os.walk(startpath):
            branches = [startpath]
            if root != startpath:
                branches.extend(os.path.relpath(root, startpath).split('/'))
            files_ = []
            for f in files:
                video = Video(os.path.join(root, f))
                id = f.split('.')[0]
                id_to_vid[id] = video
                files_.append((id, video))
                bar.update(1)
            leaf = dict([(d, {}) for d in dirs] + files_)
            set_leaf(tree, branches, leaf, bar)
        return tree[vtt_folder], id_to_vid

    def assign_features(self,
                        save_path: TextIO = 'videos_with_features.pickle'):
        for video in tqdm(self.id_to_vid.values()):
            analysis = video.analysis
            analysis_text = analysis.text
            words = video.words
            i = 0
            mid_word, last_token = False, ''
            token_to_caption_word = {}
            for token in analysis:
                if analysis_text[token.idx - 1] == ' ' and not mid_word or token.idx == 0:
                    words[i].pos = token.pos_
                    words[i].dep = token.dep_
                    words[i].head = token.head.i
                    i += 1
                else:
                    # To create a one-to-one mapping between Spacy tokens and caption words
                    # because Spacy splits tokens
                    mid_word = True
                    words[i - 1].pos += ('+' + token.pos_)
                    words[i - 1].dep += ('+' + token.dep_)
                    last_token += analysis[token.i - 1].text
                    if last_token + token.text == words[i - 1].text:
                        mid_word, last_token = False, ''
                token_to_caption_word[token.i] = i - 1
            
            for word in words:
                word.head = token_to_caption_word[word.head]

            assert i == len(words)
        self.save(save_path)


    @staticmethod
    def load(path: TextIO) -> SubtitleReader:
        print('\nLoading the videos...')
        with open(path, 'rb') as f:
            videos = pickle.load(f)
        print('Done loading.')
        return videos

    def save(self, path: TextIO) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def __getitem__(self, video_id: str) -> Video:
        return self.id_to_vid[video_id]

    def __len__(self) -> int:
        return len(self.videos)

    def nested_iter(self, value) -> Generator[Tuple[str, Video]]:
        for k, v in value.items():
            if isinstance(v, abc.Mapping):
                yield from self.nested_iter(v)
            else:
                yield k, v

    def __iter__(self) -> Generator[Tuple[str, Video]]:
        return self.nested_iter(self.videos)


def main():
    # vtt_folder = '/hd2/data/cennet/impress/data/raw/YouCookII/youcook_vtt'
    vtt_folder = '/hd2/xaines/videos/subtitles/video_ids'
    save_path = '/home/cayralat/xaines/videos_with_features.pickle'
    videos = SubtitleReader.load(save_path)
    # videos = SubtitleReader(vtt_folder, save_path)

    # counters_videos = analyze_pos_dep(videos)
    # counter_eng_sample = analyze_pos_dep_english_sample()
    # analysis = get_proportions_of_features(counters_videos, counter_eng_sample)
    # write_to_csv(analysis)
    # get_subtree(videos, feature='pos', value='VERB')
    # body_parts, proportion = body_parts_counts(videos)
    # get_body_parts_and_contexts(videos)

    for video in videos.id_to_vid.values():
        contexts = get_contexts(video)
    

    


if __name__ == '__main__':
    main()
