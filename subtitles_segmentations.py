from __future__ import annotations
from collections import abc, Counter
from tqdm import tqdm
import pickle
import os
from typing import TextIO, Dict, Generator, Tuple

from helpers import Video, Caption, Word, model
from utils import *

class SubtitleReader:
    def __init__(self,
                 vtt_folder: TextIO,
                 save_path: TextIO = '') -> None:
        """This class parses vtt subtitle files and stores them in an organized dictionary
        the hierarchy of which is the same as the directory hierarchy in which the subtitle
        files are stored. It returns a `SubtitlesReader` object which contains the parsed subtitles
        in a structured way, and with syntactic dependencies and POS for each word.

        Args:
            vtt_folder (TextIO): Path of the directory in which the subfolders are stored.
            save_path (TextIO, optional): Path of the pickle file to which we should save the
            subtitles object. Defaults to ''.
        """
        self.vtt_folder = vtt_folder
        self.videos, self.id_to_vid = self.read_videos(vtt_folder)
        self.videos.assign_features(save_path)

        if save_path:
            self.save(save_path)

    def read_videos(self, vtt_folder: TextIO) -> Tuple[Dict, Dict[str, Video]]:
        """This method reads the subtitle files and stores them in a nested dictionaries
        the hierarchy of which is the same as the subfolders hierarchy.

        Args:
            vtt_folder (TextIO): Path of the directory in which the subfolders are stored.

        Returns:
            Tuple[Dict, Dict]: The first dictionary contains nested dictionaries based on the
            structure of the subfolders. The second dictionary is a simple mapping between the
            videos and their IDs (there are no nested dictionaries).
        """

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
                        save_path: TextIO = 'videos_with_features.pickle') -> None:
        """This method runs the `spaCy` pipeline on each subtitle file which includes
        syntactic relations (`dep_` and `head`) and POS tagging (`pos_`). We only store
        the mentioned features because storing the whole analysis for each file would
        require a lot of memory.

        Args:
            save_path (TextIO, optional): Path of the `pickle` file to which we should save the
            subtitles object. Defaults to 'videos_with_features.pickle'.
        """
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
                    # because Spacy splits tokens (e.g., don't -> do + n't)
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
        """This method should be used if we have already parsed the subtitles and have
        stored them in a `pickle` file from which we want to load them.

        Args:
            path (TextIO): Path of the `pickle` file from which we should load the
            subtitles object.

        Returns:
            SubtitleReader: Subtitles object which contains the parsed subtitles in a structured
            way, and with syntactic dependencies and POS for each word.
        """
        print('\nLoading the videos...')
        with open(path, 'rb') as f:
            videos: SubtitleReader = pickle.load(f)
        print('Done loading.')
        return videos

    def save(self, path: TextIO) -> None:
        """Save the `SubtitlesReader` object to a `pickle` file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def __getitem__(self, video_id: str) -> Video:
        """With the slice notation, we search the `SubtitlesReader` object by video id
        (and not by video category)."""
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
        """When the `SubtitlesReader` object is used as an iterator, the elements returned
        are a key/value tuple which are the video category (key) and the `Video` object."""
        return self.nested_iter(self.videos)


def main():
    # vtt_folder = '/hd2/data/cennet/impress/data/raw/YouCookII/youcook_vtt'
    vtt_folder = '/hd2/xaines/videos/subtitles/video_ids'
    save_path = '/home/cayralat/xaines/videos_with_features.pickle'
    videos = SubtitleReader.load(save_path)
    # videos = SubtitleReader(vtt_folder, save_path)

    # Below are a bunch of functions loaded from utils.py which use the SubtitleReader
    # class to infer some statistics from the dataset.

    # counters_videos = analyze_pos_dep(videos)
    # counter_eng_sample = analyze_pos_dep_english_sample()
    # analysis = get_proportions_of_features(counters_videos, counter_eng_sample)
    # write_to_csv(analysis)
    # get_subtree(videos, feature='pos', value='VERB')
    # body_parts, proportion = body_parts_counts(videos)
    # get_body_parts_and_contexts(videos)

    # for video in videos.id_to_vid.values():
    #     contexts = get_contexts(video)

    analyze_verb_distribution(videos)
    pass
    

    


if __name__ == '__main__':
    main()
