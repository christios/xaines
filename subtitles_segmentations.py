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

    def analyze_pos_dep(self,
                        counters_path='/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/Saarland Univeristy/Winter 2020-2021/hiwi/youtube_videos/counters_videos.pickle',
                        reset=False):
        counters = {
            'fights': {'dep': Counter(), 'pos': Counter(), 'dep_pos': Counter()},
            'dances': {'dep': Counter(), 'pos': Counter(), 'dep_pos': Counter()},
            'spots': {'dep': Counter(), 'pos': Counter(), 'dep_pos': Counter()}
        }
        if reset or not os.path.exists(counters_path):
            for category, content in self.videos.items():
                print(category)
                sub_cat_samples = 30 // len(content)
                for subcategory, sub_content in content.items():
                    print(subcategory)
                    subcat_counter = 0
                    for i, (video_id, video) in enumerate(sub_content.items()):
                        print(video_id)
                        analysis = video.analysis
                        counters[category]['dep'].update(
                            [token.dep_ for token in analysis])
                        counters[category]['pos'].update(
                            [token.pos_ for token in analysis])
                        counters[category]['dep_pos'].update(
                            [(token.dep_, token.pos_) for token in analysis])
                        subcat_counter += 1
                        if i > sub_cat_samples:
                            break
                    if subcat_counter > 30:
                        break
            with open(counters_path, 'wb') as f:
                pickle.dump(counters, f)
        else:
            with open(counters_path, 'rb') as f:
                counters = pickle.load(f)
        return counters

    def get_subtree(self, feature, value):
        for video_id, video in iter(self):
            print(f'Showing sentences from video with ID {video_id}')
            analysis = video.analysis
            analysis_iter = iter(analysis)
            i = 0
            while True:
                try:
                    token = next(analysis_iter)
                    i += 1
                    if getattr(token, feature + '_') == value:
                        print()
                        print(' '.join([token.text for token in token.subtree]))
                        inp = input('\nPress Enter to see next sentence or c + Enter to see some context: ')
                        if inp == 'c':
                            print()
                            print('[...]', ' '.join([token.text for token in analysis[max(0,i-10):i+10]]), '[...]')
                            print('HEAD: ', token.head.text)
                            inp = input(
                                '\nPress Enter to see next sentence: ')
                except StopIteration:
                    break

    def body_parts_counts(self) -> Tuple[Counter, float]:
        body_parts = Counter()
        total_words = 0
        for video in self.id_to_vid.values():
            total_words += len(video)
            for word in video.words:
                if word.pos == 'NOUN' and word.text in BODY_PARTS:
                    body_parts.update([word.text])
        numer_body_parts = sum(n for n in body_parts.values())
        nouns = total_words * 0.16
        # 30% of nouns in our data are body parts
        proportion = numer_body_parts/nouns
        return body_parts, proportion


    def get_body_parts_and_contexts(self):
        for video in tqdm(self.id_to_vid.values()):
            contexts = []
            for i, word in enumerate(video.words):
                if word.pos == 'NOUN' and word.text in BODY_PARTS:
                    contexts.append([])
                    for x in range(3, 6):
                        contexts[-1].append(
                            ' '.join([w.text for w in video.words[max(0, i - x): i + x]]))
            head, tail = os.path.split(video.file_path)
            if not os.path.isdir(os.path.join(head, 'bp')):
                os.mkdir(os.path.join(head, 'bp'))
            with open(os.path.join(head, 'bp', tail.split('.')[0]), 'w') as f:
                for x in range(3, 6):
                    print(f'context-{x}', file=f, end='\t')
                print(file=f)
                for context in contexts:
                    for x in context:
                        print(x, file=f, end='\t')
                    print(file=f)

def main():
    # vtt_folder = '/hd2/data/cennet/impress/data/raw/YouCookII/youcook_vtt'
    vtt_folder = '/hd2/xaines/videos/subtitles/video_ids'
    save_path = '/home/cayralat/xaines/videos_with_features.pickle'
    videos = SubtitleReader.load(save_path)
    # videos = SubtitleReader(vtt_folder, save_path)

    # counters_videos = videos.analyze_pos_dep()
    # counter_eng_sample = analyze_pos_dep_english_sample()
    # analysis = get_proportions_of_features(counters_videos, counter_eng_sample)
    # write_to_csv(analysis)
    # videos.get_subtree('pos', 'VERB')
    # body_parts, proportion = videos.body_parts_counts()
    # videos.get_body_parts_and_contexts()

    for video in videos.id_to_vid.values():
        contexts = video.get_contexts()
    

    


if __name__ == '__main__':
    main()
