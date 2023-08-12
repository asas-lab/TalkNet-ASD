## Download videos

The 'download.py' script can be used for quick testing, it downloads the video directly from its URL.

```
python download.py --videoList test.csv
```

The 'download_2.py' script can be used for the formal format of the CSV file, it downloads all videos for each playlist URL of podcast channels.

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
		├── pyavi
		│   ├── audio.wav (Audio from input video)
		│   ├── video.avi (Copy of the input video)
		│   ├── video_only.avi (Output video without audio)
		│   └── video_out.avi  (Output video with audio)
		├── pycrop (The detected face videos and audios)
		│   ├── 000000.avi
		│   ├── 000000.wav
		│   ├── 000001.avi
		│   ├── 000001.wav
		│   └── ...
		└── pywork
	    	├── faces.pckl (face detection result)
	    	├── scene.pckl (scene detection result)
	    	├── scores.pckl (ASD result)
	    	└── tracks.pckl (face tracking result)
```

