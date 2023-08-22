## Download videos

The 'download.py' script can be used for the formal format of the CSV file, it downloads all videos for each playlist URL of podcast channels.

```
python download_2.py --videoList podcasts_list.csv
```

## All stages (scenes detection, face detection, face tracking, face clips cropping, active speaker detection)

The 'ASD.py' script can be used to do the remaining stages

```
python ASD.py 
```

---

The output files structure is illustrated below:

```
└── videos
	├── podcast_# (folder for each podcast)
	│   ├── videoIDd.mp4 (all videos in the podcast)
	│   ├── metadata (folder to store metadata for each video)
	│   	├── videoIDd.josn (josn file for all videos)
│
└── output
	├── podcast_# (folder for each podcast)
	│   ├── videoIDd (folder for each video on the podcast)
		├── pycrop (The talking head videos and audios)
		│   ├── 000000.avi
		│   ├── 000000.wav
		│   ├── 000001.avi
		│   ├── 000001.wav
		│   └── ...
```

