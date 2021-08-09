from typing import List, Optional, TextIO, Tuple, Generator
import os
import re
import datetime

import webvtt

import spacy

pattern_time = r"<(.[0-9:.]+)>"
pattern_word = r"<c>(.*?)</c>"
pattern_first = r".*?<[0-9:.]+>"
pattern_rest = r"<[0-9:.]+><c>\s.*?</c>"

try:
    model = spacy.load("en_core_web_trf")
except:
    model = None
    print('The TRF Spacy model is not available. Either download it using:\n\n\tpython -m spacy download en_core_web_trf\n\nor do not use any function which might need it.')

class Word:
    def __init__(self,
                 text: str,
                 start: Optional[float] = None,
                 end: Optional[float] = None,
                 pos: Optional[str] = None,
                 dep: Optional[str] = None,
                 head: Optional[int] = None) -> None:
        self.text = text
        self.start = start
        self.end = end
        self.pos = pos
        self.dep = dep
        self.head = head

    def __repr__(self) -> str:
        return f"({self.start}, {self.text}, {self.end})"

    def __add__(self, other: 'Word') -> str:
        return self.text + other.text


class Caption:
    def __init__(self,
                 is_word_aligned: bool,
                 start: Optional[float] = None,
                 end: Optional[float] = None) -> None:
        self.words: List[Word] = []
        self.is_word_aligned = is_word_aligned
        self.start = start
        self.end = end

    def append(self, word: Word) -> None:
        self.words.append(word)

    def __getitem__(self, index) -> Word:
        return self.words[index]

    def __repr__(self) -> str:
        return ' '.join([word.text for word in self.words])

    def __bool__(self) -> bool:
        return bool(self.words)

    def __len__(self) -> int:
        return len(self.words)

    def __delitem__(self, index) -> None:
        del self.words[index]


class Video:
    def __init__(self,
                 file_path: TextIO) -> None:
        self.file_path = file_path
        self.file_name = os.path.basename(
            self.file_path.replace('.en.vtt', ''))
        self.captions = Video._parse_vtt_file(file_path)
        self.preprocess()

    def preprocess(self):
        for caption in self.captions:
            words_ = []
            for word in caption:
                if ' ' in word.text:
                    split_words = []
                    split = word.text.split()
                    n = len(split)
                    timestamps_diff = word.end - word.start
                    timestamps = [timestamps_diff *
                                  i / n for i in range(n + 1)]
                    for i in range(n):
                        word_ = Word(text=split[i],
                                     start=timestamps[i],
                                     end=timestamps[i + 1])
                        split_words.append(word_)
                    words_ += split_words
                else:
                    words_.append(word)
            caption.words = words_

    def __len__(self) -> int:
        return sum(len(caption) for caption in self.captions)

    def __repr__(self) -> str:
        return ' '.join([word.text for caption in self.captions for word in caption])

    def __str__(self) -> str:
        return ' '.join([word.text for caption in self.captions for word in caption])

    def nested_iter(self, captions) -> Generator[Word, None, None]:
        for caption in captions:
            for word in caption:
                yield word

    def __iter__(self) -> Generator[Word, None, None]:
        return self.nested_iter(self.captions)

    @property
    def words(self) -> List[Word]:
        return [word for caption in self.captions for word in caption]

    @property
    def analysis(self):
        return model(str(self))

    @property
    def sentences(self):
        return list(self.analysis.sents)


    @staticmethod
    def visualize_dependency_tree(sentence) -> None:
        spacy.displacy.serve(sentence, style="dep")

    @staticmethod
    def _parse_vtt_file(file_path) -> List[Caption]:
        captions: List[Caption] = []
        for file_caption in webvtt.read(file_path):
            for line in file_caption.lines:
                if '<c>' not in line:
                    continue
                caption = Caption(is_word_aligned=True,
                                  start=file_caption.start_in_seconds,
                                  end=file_caption.end_in_seconds)

                first = re.match(pattern_first, line, re.M | re.I)
                first_word, start = Video._remove_tags(first[0])
                # Estimate the time here because it is not given in the vtt file
                if captions:
                    junction = round((start + captions[-1][-1].start) / 2, 3)
                    if junction > 1:
                        junction = start - 1
                    captions[-1][-1].end = junction
                    first_word = Word(text=first_word, start=junction)
                else:
                    first_word = Word(
                        text=first_word,
                        start=max(round(start - 1, 3), 0))
                caption.append(first_word)

                rest = re.findall(pattern_rest, line, re.M | re.I)
                for match in rest:
                    next_word, start = Video._remove_tags(match)
                    if len(caption) == 1 and not re.search(r'\w', caption[0].text):
                        del caption[0]
                    else:
                        caption[-1].end = start
                    next_word = Word(text=next_word, start=start)
                    caption.append(next_word)
                captions.append(caption)
                break
        # If file is not word aligned with video
        if not captions:
            for file_caption in webvtt.read(file_path):
                caption = Caption(is_word_aligned=False,
                                  start=file_caption.start_in_seconds,
                                  end=file_caption.end_in_seconds)
                for word in file_caption.raw_text.split():
                    caption.append(Word(word.strip()))
                if caption:
                    caption[0].start = file_caption.start_in_seconds
                    caption[-1].end = file_caption.end_in_seconds
                    captions.append(caption)
        else:
            # Also here, estimate the time which is not given in the vtt file
            captions[-1][-1].end = round(captions[-1][-1].start + 1, 3)

        return captions

    @staticmethod
    def _hour2second(time: str) -> float:
        date_time = datetime.datetime.strptime(time, "%H:%M:%S.%f")
        a_timedelta = date_time - datetime.datetime(1900, 1, 1)
        second = a_timedelta.total_seconds()
        return second

    @staticmethod
    def _remove_tags(word_tag: str) -> Tuple[str, float]:
        # it can be time or <c> tag or </c> tag
        if '<c>' in word_tag:
            w = re.search(pattern_word, word_tag, flags=0)
            t = re.search(pattern_time, word_tag, flags=0)
            return w[1].strip(), Video._hour2second(t[1])
        else:
            t = re.search(pattern_time, word_tag, flags=0)
            w = re.sub(pattern_time, '', word_tag)
            return w.strip(), Video._hour2second(t[1])

    def to_vtt_format(self,
                      attr: str = 'raw_text') -> None:
        """attr: {start, end, text, raw_text, identifier}"""
        for i, caption in enumerate(webvtt.read(self.file_path)):
            print('------')
            print(attr.upper())
            print(getattr(caption, attr))
