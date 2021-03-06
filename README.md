# XAINES Project

This repository contains a script which allows you to read subtitles from *vtt* files and store them in an organized dictionary the hierarchy of which is based on the directory hierarchy in which the subtitle files are stored.

## Instantiation

The main python file is `subtitles_segmentations.py`. To parse and analyze the data, run:

    python3 subtitles_segmentations.py

The class which allows you to parse subtitles is `SubtitleReader`. The first way of instantiating an object is to call the constructor of the class, in which case the subtitle files in the provided path will be parsed. This takes quite some time (around 10 minutes):
    
    videos = SubtitleReader(vtt_folder, save_path)

The other way is to load a pickle object generated by the latter:

    videos = SubtitleReader.load(save_path)

where:
- `vtt_folder` is the folder which contains all video category subfolders
- `save_path` is the path of the pickle file which will be created for the parsed videos

## Data Manipulation

Essentially, the `SubtitleReader` object (`video`) is a list of `Caption` objects, which are in turn lists of `Word` objects. There are two main ways to interact with this object.

### Accessing by ID

You can access videos using their IDs:

    video = videos['3miio1kz3gc']

You can loop through the videos in the following way:

    for video_id, video in iter(videos):
        pass

or

    for video_id, video in videos.id_to_vid.items():
        pass

### Accessing by Category

Finally, one can access the videos by their categories:

    video = videos.videos['fights']['capoeira_beginners']['3miio1kz3gc']

You can also loop through the videos with their associated categories:

    for cat_name, category in videos.videos.items():
        for subcat_name, subcategory in category.items():
            for video_id, video in subcategory.items():
                pass

### Video Object

There are multiple ways to interact with the `video` object:

    video_words = video.words # List of all `Word` objects in the video
    captions = video.captions # List of all `Caption` objects in the video
    caption_words = captions[0].words # List of all `Word` objects in the first caption of the video
    caption_word = caption_words[0] # Gets first `Word` object in the first caption
    video_analysis = video.analysis # Runs a spaCy analysis on the video and returns it
    video_sentences = video.sentences # Runs a spaCy analysis on the video and segments the words into sentences

