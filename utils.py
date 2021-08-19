from collections import Counter
from tqdm import tqdm
import pickle
import os
import csv
from typing import Tuple

import numpy as np

from helpers import Video, Caption, Word, model
import subtitles_segmentations as ss

BODY_PARTS = ['arm', 'eye', 'eyebrow', 'belly', 'leg', 'breast', 'thumb', 'elbow',
              'finger', 'foot', 'ankle', 'buttocks', 'hair', 'neck', 'face',
              'hand', 'wrist', 'hip', 'chin', 'knee', 'head', 'lip', 'mouth', 'nose',
              'nostril', 'thigh', 'ear', 'bottom', 'bum', 'back', 'underarm', 'forearm',
              'leg', 'shoulder', 'forehead', 'waist', 'calf', 'cheek',
              'eyelash', 'lash', 'tooth', 'toe', 'tongue', 'muscle', 'lung', 'spine', 'stomach',
              'chest', 'abdominal', 'ab', 'hamstring', 'quadricep', 'quad', 'glute', 'bicep',
              'tricep', 'forearm', 'trap']
BODY_PARTS_PLURAL = ['feet', 'calves', 'teeth'] + \
    [bp + 's' for bp in BODY_PARTS]
BODY_PARTS += BODY_PARTS_PLURAL
BODY_PARTS = {bp: None for bp in BODY_PARTS}

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


def analyze_verb_distribution(videos,
                              analysis='tf_idf',
                              verb_counters_path='/home/cayralat/xaines/verb_counters.pickle'):
    if os.path.exists(verb_counters_path):
        with open(verb_counters_path, 'rb') as f:
            counters = pickle.load(f)
    else:
        counters = {}
        for category, content in videos.videos.items():
            counters[category] = dict()
            for subcategory, sub_content in content.items():
                counters[category][subcategory] = Counter()
                for video_id, video in sub_content.items():
                    counters[category][subcategory].update(
                        [word.text for word in video if word.pos == 'VERB'])
                counters[category][subcategory] = counters[category][subcategory].most_common()
        with open(verb_counters_path, 'wb') as f:
            pickle.dump(counters, f)

    if analysis == 'tf_idf':
        tf_idf_matrix = tf_idf(counters)

    elif analysis == 'distribution':
        verb_distribution_per_category(counters)

    return counters


def verb_contexts_distribution(videos: ss.SubtitleReader,
                               window=2):
    with tqdm(total=len(videos.id_to_vid)) as progress_bar:
        verb_contexts = {}
        for category, content in videos.videos.items():
            for subcategory, sub_content in content.items():
                for video in sub_content.values():
                    words = video.words
                    for i, word in enumerate(words):
                        if word.pos and 'VERB' in word.pos:
                            start = max(0, i - window)
                            stop = i + window + 1
                            context = tuple([(w.pos, w.text) for w in words[start:stop]])
                            verb_contexts.setdefault(subcategory, []).append(context)
                    progress_bar.update(1)
    verb_contexts_dist = {}
    for category, contexts in verb_contexts.items():
        value = verb_contexts_dist.setdefault(category, {'pos': Counter()})
        value['pos'].update([tuple([c[0] for c in ctx]) for ctx in contexts])
        value['pos'] = value['pos'].most_common()
        value['examples'] = {}
        for ctx in contexts:
            value['examples'].setdefault(tuple([c[0] for c in ctx]), []).append(
                tuple([c[1] for c in ctx]))
        value['examples'] = {ctx: value['examples'][ctx] for ctx, _ in value['pos']}
    return verb_contexts_dist


def tf_idf(verb_counters):
    num_of_categories = sum(len(content) for content in verb_counters.values())
    verbs_fl = Counter([term[0] for content in verb_counters.values()
                       for subcontent in content.values() for term in subcontent])
    id2cat = [subcategory for content in verb_counters.values()
              for subcategory in content]
    cat2id = {cat: i for i, cat in enumerate(id2cat)}
    id2term = [term for term in verbs_fl]
    term2id = {term: i for i, term in enumerate(id2term)}
    df = np.zeros((num_of_categories, len(verbs_fl)), dtype=np.uint8)
    for category in verb_counters.values():
        for name, doc in category.items():
            for term in doc:
                x, y = cat2id[name], term2id[term[0]]
                df[x][y] = 1
    tf_idf_matrix = np.zeros((num_of_categories, len(verbs_fl)))
    for category in verb_counters.values():
        for name, doc in category.items():
            doc_total_verb_count = sum(term[1] for term in doc)
            for term in doc:
                x, y = cat2id[name], term2id[term[0]]
                # tf
                tf_idf_matrix[x][y] = term[1] / doc_total_verb_count
                # idf
                tf_idf_matrix[x][y] *= np.log10(len(id2cat) /
                                                min(df[:, y].sum(), len(id2cat) - 1))
    most_important_words = {}
    for i, category_counts in enumerate(tf_idf_matrix):
        for j, index in enumerate(np.flip(category_counts.argsort())):
            if j == 30:
                break
            most_important_words.setdefault(id2cat[i], []).append(id2term[index])

                 
    return tf_idf_matrix

def verb_distribution_per_category(counters):
    distribution = {}
    for name, category in counters.items():
        distribution.setdefault(name, [])
        c = 0
        for subcategory in category.values():
            for verb in subcategory:
                if c > 30:
                    continue
                distribution[name].append(verb)
                c += 1
    for cat in distribution:
        with open(f'/home/cayralat/xaines/{cat}_counts.tsv', 'w') as f:
            for x in distribution[cat]:
                print(*x, file=f, sep='\t')


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
        for category, content in videos.videos.items():
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
