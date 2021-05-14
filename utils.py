from collections import Counter
from tqdm import tqdm
import pickle
import os
import csv

import spacy

model = spacy.load("en_core_web_trf")

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
