from collections import Counter
from tqdm import tqdm
import pickle
import os
import csv
from typing import Tuple

from helpers import model

BODY_PARTS = ['arm', 'eye', 'eyebrow', 'belly', 'leg', 'breast', 'thumb', 'elbow',
              'finger', 'foot', 'ankle', 'buttocks', 'hair', 'neck',
              'hand', 'wrist', 'hip', 'chin', 'knee', 'head', 'lip', 'mouth', 'nose',
              'nostril', 'thigh', 'ear', 'bottom', 'bum', 'back', 'underarm', 'forearm',
              'leg', 'shoulder', 'forehead', 'waist', 'calf', 'cheek',
              'eyelash', 'lash', 'tooth', 'toe', 'tongue']
BODY_PARTS_PLURAL = ['feet', 'calves', 'teeth'] + \
    [bp + 's' for bp in BODY_PARTS]
BODY_PARTS += BODY_PARTS_PLURAL

def analyze_pos_dep_english_sample(counters_path='/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/Saarland Univeristy/Winter 2020-2021/hiwi/youtube_videos/counters_eng_sample.pickle',
                                   reset=False):
    counters = {'dep': Counter(), 'pos': Counter(), 'dep_pos': Counter()}
    if reset or not os.path.exists(counters_path):
        with open('eng_sample.txt') as f, open('counters_eng_sample.pickle', 'wb') as f_out:
            corpus = f.readlines()
            for line in tqdm(corpus[:len(corpus)//2]):
                analysis = model(line)
                counters['dep'].update(
                    [token.dep_ for token in analysis])
                counters['pos'].update(
                    [token.pos_ for token in analysis])
                counters['dep_pos'].update(
                    [(token.dep_, token.pos_) for token in analysis])
            pickle.dump(counters, f_out)
    else:
        with open(counters_path, 'rb') as f:
            counters = {'eng_sample': pickle.load(f)}
    return counters


def get_proportions_of_features(counters_videos, counter_eng_sample):
    analysis = {}
    for category, cat_counters in {**counters_videos, **counter_eng_sample}.items():
        analysis[category] = {}
        for f, f_counter in cat_counters.items():
            total = sum(count for count in f_counter.values())
            analysis[category][f] = sorted({value: round(count/total, 2) for value, count in f_counter.items(
            ) if count/total > 0.01}.items(), key=lambda x: x[1], reverse=True)
    return analysis


def get_subtree(videos, feature, value):
    for video_id, video in iter(videos):
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
                    print(
                        ' '.join([token.text for token in token.subtree]))
                    inp = input(
                        '\nPress Enter to see next sentence or c + Enter to see some context: ')
                    if inp == 'c':
                        print()
                        print('[...]', ' '.join(
                            [token.text for token in analysis[max(0, i-10):i+10]]), '[...]')
                        print('HEAD: ', token.head.text)
                        inp = input(
                            '\nPress Enter to see next sentence: ')
            except StopIteration:
                break


def body_parts_counts(videos) -> Tuple[Counter, float]:
    body_parts = Counter()
    total_words = 0
    for video in videos.id_to_vid.values():
        total_words += len(video)
        for word in video.words:
            if word.pos == 'NOUN' and word.text in BODY_PARTS:
                body_parts.update([word.text])
    numer_body_parts = sum(n for n in body_parts.values())
    nouns = total_words * 0.16
    # 30% of nouns in our data are body parts
    proportion = numer_body_parts/nouns
    return body_parts, proportion


def get_body_parts_and_contexts(videos):
    for video in tqdm(videos.id_to_vid.values()):
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


def analyze_pos_dep(videos,
                    counters_path='/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/Saarland Univeristy/Winter 2020-2021/hiwi/youtube_videos/counters_videos.pickle',
                    reset=False):
    counters = {
        'fights': {'dep': Counter(), 'pos': Counter(), 'dep_pos': Counter()},
        'dances': {'dep': Counter(), 'pos': Counter(), 'dep_pos': Counter()},
        'spots': {'dep': Counter(), 'pos': Counter(), 'dep_pos': Counter()}
    }
    if reset or not os.path.exists(counters_path):
        for category, content in videos.items():
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


def get_contexts(video):
    head, tail = os.path.split(video.file_path)
    contexts = {}
    with open(os.path.join(head, 'bp', tail.split('.')[0])) as f:
        for i, line in enumerate(f):
            if i == 0:
                for c in line.strip().split('\t'):
                    contexts[c] = []
            else:
                for j, c in enumerate(line.strip().split('\t')):
                    contexts['context-' + str(j + 3)].append(c)
    return contexts


def write_to_csv(analysis):
    features = {'dep': set(), 'pos': set(), 'dep_pos': set()}
    for counters in analysis.values():
        for f, statistics in counters.items():
            features[f].update([s[0] for s in statistics])
    features = {feature: list(values) for feature, values in features.items()}

    csv_file = []
    for counters in analysis.values():
        for f in counters.keys():
            for value in features[f]:
                csv_file.append([value])
        break
    for corpus_type, counters in analysis.items():
        i = 0
        for f, statistics in counters.items():
            statistics = dict(statistics)
            for value in features[f]:
                csv_file[i].append(statistics.get(value))
                i += 1
    csv_file.insert(0, ['feature'] + [dataset for dataset in analysis.keys()])

    with open("results1.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_file)
