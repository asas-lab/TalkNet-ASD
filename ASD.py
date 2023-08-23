import sys, time, os, tqdm, torch, json, argparse, glob, subprocess, warnings, cv2, numpy, math, python_speech_features

import pandas as pd
from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector


from model.faceDetector.s3fd import S3FD
from talkNet import talkNet

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description = "TalkNet ASD")
parser.add_argument('--videoFolder',           type=str, default="videos",  help='Path for inputs, tmps and outputs')
parser.add_argument('--outputFolder', 	       type=str, default="output", help='Path for output of ASD')
parser.add_argument('--pretrainModel',         type=str, default="pretrain_TalkSet.model",   help='Path for the pretrained TalkNet model')

parser.add_argument('--nDataLoaderThread',     type=int,   default=10,   help='Number of workers')
parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
parser.add_argument('--minTrack',              type=int,   default=10,   help='Number of min frames for each shot')
parser.add_argument('--numFailedDet',          type=int,   default=10,   help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')
parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')

parser.add_argument('--start',                 type=int, default=0,   help='The start time of the video')
parser.add_argument('--duration',              type=int, default=0,  help='The duration of the video, when set as 0, will extract the whole video')

args = parser.parse_args()

if os.path.isfile(args.pretrainModel) == False: # Download the pretrained model
    Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
    cmd = "gdown --id %s -O %s"%(Link, args.pretrainModel)
    subprocess.call(cmd, shell=True, stdout=None)


def write_json(new_data, filename, flag):
	# CPU: store the metadata
	with open(filename, "r", encoding='utf-8') as jsonFile:
		data = json.load(jsonFile)
	if flag == 0: # first clip
		data["clips"] = new_data
	else: # remaining clips
		data["clips"].update(new_data)

	with open(filename, "w", encoding='utf-8') as jsonFile:
		json.dump(data, jsonFile,ensure_ascii=False, indent=4,separators=(',', ': '))


def update_json(nonTalkingClips, filename):
	# CPU: store the metadata
	with open(filename, "r", encoding='utf-8') as jsonFile:
		data = json.load(jsonFile)

	for clipId in nonTalkingClips:
		del data["clips"][clipId] # delete non-talking clips info

	with open(filename, "w", encoding='utf-8') as jsonFile:
		json.dump(data, jsonFile,ensure_ascii=False, indent=4,separators=(',', ': '))


def scene_detect(args):
	# CPU: Scene detection, output is the list of each shot's time duration
	videoManager = VideoManager([args.videoFilePath])
	statsManager = StatsManager()
	sceneManager = SceneManager(statsManager)
	sceneManager.add_detector(ContentDetector())
	baseTimecode = videoManager.get_base_timecode()
	videoManager.set_downscale_factor()
	videoManager.start()
	sceneManager.detect_scenes(frame_source = videoManager)
	sceneList = sceneManager.get_scene_list(baseTimecode)
	if sceneList == []:
		sceneList = [(videoManager.get_base_timecode(),videoManager.get_current_timecode())]

	sys.stderr.write('%s - scenes detected %d\n'%(args.videoFilePath, len(sceneList)))
	return sceneList

def inference_video(args):
	# GPU: Face detection, output is the list contains the face location and score in this frame
	DET = S3FD(device='cuda')
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	flist.sort()
	dets = []
	for fidx, fname in enumerate(flist):
		image = cv2.imread(fname)
		imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
		dets.append([])
		for bbox in bboxes:
		  dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
		sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
	return dets

def bb_intersection_over_union(boxA, boxB):
	# CPU: IOU Function to calculate overlap between two image
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def track_shot(args, sceneFaces):
	# CPU: Face tracking
	iouThres  = 0.5     # Minimum IOU between consecutive face detections
	tracks    = []
	while True:
		track     = []
		for frameFaces in sceneFaces:
			for face in frameFaces:
				if track == []:
					track.append(face)
					frameFaces.remove(face)
				elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
					iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
					if iou > iouThres:
						track.append(face)
						frameFaces.remove(face)
						continue
				else:
					break
		if track == []:
			break
		elif len(track) > args.minTrack:
			frameNum    = numpy.array([ f['frame'] for f in track ])
			bboxes      = numpy.array([numpy.array(f['bbox']) for f in track])
			frameI      = numpy.arange(frameNum[0],frameNum[-1]+1)
			bboxesI    = []
			for ij in range(0,4):
				interpfn  = interp1d(frameNum, bboxes[:,ij])
				bboxesI.append(interpfn(frameI))
			bboxesI  = numpy.stack(bboxesI, axis=1)
			if max(numpy.mean(bboxesI[:,2]-bboxesI[:,0]), numpy.mean(bboxesI[:,3]-bboxesI[:,1])) > args.minFaceSize:
				tracks.append({'frame':frameI,'bbox':bboxesI})
	return tracks

def crop_video(args, track, cropFile, json_path, flag, clip_id):
	# CPU: crop the face clips
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Read the frames
	flist.sort()
	vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))# Write video

	# store the metadata
	clipStart = (track['frame'][0]) / 25
	clipEnd = (track['frame'][-1]+1) / 25
	bbox_list = {}
	for idx, box in enumerate(track['bbox']):
		bbox_list[idx] =  [round(box[0]), round(box[1]), round(box[2]), round(box[3])]

	metadata = {
		str(clip_id): {
			"duration" : {"start_sec": clipStart, "end_sec": clipEnd},
			"bbox" : bbox_list
		}
	}
	write_json(metadata, json_path, flag)
	
	dets = {'x':[], 'y':[], 's':[]}
	for det in track['bbox']: # Read the tracks
		dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
		dets['y'].append((det[1]+det[3])/2) # crop center x 
		dets['x'].append((det[0]+det[2])/2) # crop center y
	dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
	dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
	dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
	for fidx, frame in enumerate(track['frame']):
		cs  = args.cropScale
		bs  = dets['s'][fidx]   # Detection box size
		bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
		image = cv2.imread(flist[frame])
		frame = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
		my  = dets['y'][fidx] + bsi  # BBox center Y
		mx  = dets['x'][fidx] + bsi  # BBox center X
		face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
		vOut.write(cv2.resize(face, (224, 224)))
	audioTmp    = cropFile + '.wav'
	audioStart  = (track['frame'][0]) / 25
	audioEnd    = (track['frame'][-1]+1) / 25
	vOut.release()
	command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
		      (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp)) 
	output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
	_, audio = wavfile.read(audioTmp)
	command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
			  (cropFile, audioTmp, args.nDataLoaderThread, cropFile)) # Combine audio and video file
	output = subprocess.call(command, shell=True, stdout=None)
	os.remove(cropFile + 't.avi')
	return {'track':track, 'proc_track':dets}


def evaluate_network(files, args, json_path):
	# GPU: active speaker detection by pretrained TalkNet
	s = talkNet()
	s.loadParameters(args.pretrainModel)
	sys.stderr.write("Model %s loaded from previous state! \r\n"%args.pretrainModel)
	s.eval()
	allScores = []
	# durationSet = {1,2,4,6} # To make the result more reliable
	durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
	nonTalkingClips = [] # To store the ids for non-talking clips
	for file in tqdm.tqdm(files, total = len(files)):
		fileName = os.path.splitext(file.split('/')[-1])[0] # Load audio and video
		_, audio = wavfile.read(fileName + '.wav')
		audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
		video = cv2.VideoCapture(fileName + '.avi')
		videoFeature = []
		while video.isOpened():
			ret, frames = video.read()
			if ret == True:
				face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
				face = cv2.resize(face, (224,224))
				face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
				videoFeature.append(face)
			else:
				break
		video.release()
		videoFeature = numpy.array(videoFeature)
		length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0] / 25)
		audioFeature = audioFeature[:int(round(length * 100)),:]
		videoFeature = videoFeature[:int(round(length * 25)),:,:]
		allScore = [] # Evaluation use TalkNet
		for duration in durationSet:
			batchSize = int(math.ceil(length / duration))
			scores = []
			with torch.no_grad():
				for i in range(batchSize):
					inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()
					inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()
					embedA = s.model.forward_audio_frontend(inputA)
					embedV = s.model.forward_visual_frontend(inputV)	
					embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
					out = s.model.forward_audio_visual_backend(embedA, embedV)
					score = s.lossAV.forward(out, labels = None)
					scores.extend(score)
			allScore.append(scores)
		allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
		averageFramesScores = numpy.mean(allScore)

		# remove non-talking clips
		if (averageFramesScores < 0): # non-talking clip if the average frames score is less than zero
			clipId = str(fileName.split('pycrop')[1][1:])
			nonTalkingClips.append(clipId)
			os.remove(fileName + '.wav')
			os.remove(fileName + '.avi')

		allScores.append(allScore)
	update_json(nonTalkingClips, json_path) # remove non-talking clips info
	return allScores


# Main function
def main():

	videosPath = args.videoFolder
	os.makedirs(args.outputFolder, exist_ok = True) # Save the results in in this path


	# loop for each podcast list
	podcast_list = os.listdir(videosPath)
	for podcast in podcast_list:
		# directory output for current podcast
		output_podcast = os.path.join(args.outputFolder, podcast)
		if not os.path.exists(output_podcast):
			os.makedirs(output_podcast)

		podcast_path = os.path.join(videosPath, podcast)

		# loop for each video belong to current podcast
		videos_list = os.listdir(podcast_path)
		for video_path in videos_list:
			# complete directory scenes
			output_podcast_video = os.path.join(
				output_podcast, video_path.split('.')[0])
			if not os.path.exists(output_podcast_video):
				os.makedirs(output_podcast_video)

			# Initialization
			args.videoPath = os.path.join(podcast_path, video_path)
			args.savePath = output_podcast_video

			video_id = video_path.split('.')[0]
			json_path = "videos/"+ podcast +"/metadata/" + video_id + ".json"

			args.pyaviPath = os.path.join(args.savePath, 'pyavi')
			args.pycropPath = os.path.join(args.savePath, 'pycrop')
			args.pyframesPath = os.path.join(args.savePath, 'pyframes')

			if os.path.exists(args.savePath):
				rmtree(args.savePath)

			os.makedirs(args.pyaviPath, exist_ok=True)  # The path for the input video, input audio, output video
			os.makedirs(args.pycropPath, exist_ok=True)  # Save the detected face clips (audio+video) in this process
			os.makedirs(args.pyframesPath, exist_ok=True)  # Save all the video frames

			# Extract video
			args.videoFilePath = os.path.join(args.pyaviPath, 'video.avi')
			# If duration did not set, extract the whole video, otherwise extract the video from 'args.start' to 'args.start + args.duration'
			if args.duration == 0:
				command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" % \
				           (args.videoPath, args.nDataLoaderThread, args.videoFilePath))
			else:
				command = (
							"ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic" % \
							(args.videoPath, args.nDataLoaderThread, args.start, args.start + args.duration,
							 args.videoFilePath))
			subprocess.call(command, shell=True, stdout=None)
			sys.stderr.write(
				time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the video and save in %s \r\n" % (args.videoFilePath))

			# Extract audio
			args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
			command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
			           (args.videoFilePath, args.nDataLoaderThread, args.audioFilePath))
			subprocess.call(command, shell=True, stdout=None)
			sys.stderr.write(
				time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" % (args.audioFilePath))

			# Extract the video frames
			command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" %
			           (args.videoFilePath, args.nDataLoaderThread, os.path.join(args.pyframesPath, '%06d.jpg')))
			subprocess.call(command, shell=True, stdout=None)
			sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
			                 " Extract the frames and save in %s \r\n" % (args.pyframesPath))

			# Scene detection for the video frames
			scene = scene_detect(args)
			sys.stderr.write(
				time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection")

			# Face detection for the video frames
			faces = inference_video(args)
			sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection")

			# Face tracking
			allTracks, vidTracks = [], []
			for shot in scene:
				if shot[1].frame_num - shot[
					0].frame_num >= args.minTrack:  # Discard the shot frames less than minTrack frames
					allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[
						1].frame_num]))  # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
			sys.stderr.write(
				time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" % len(allTracks))

			# Face clips cropping
			for ii, track in tqdm.tqdm(enumerate(allTracks), total=len(allTracks)):
				if ii == 0: # first clip
					vidTracks.append(crop_video(args, track, os.path.join(args.pycropPath, '%05d' % ii), json_path, flag=0, clip_id = '%05d' % ii))
				else:
					vidTracks.append(crop_video(args, track, os.path.join(args.pycropPath, '%05d' % ii), json_path, flag=1, clip_id = '%05d' % ii))
			sys.stderr.write(
				time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop and saved in %s tracks \r\n" % args.pycropPath)

			# Active Speaker Detection by TalkNet
			files = glob.glob("%s/*.avi" % args.pycropPath)
			files.sort()
			scores = evaluate_network(files, args, json_path)
			sys.stderr.write(
				time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted")

			# remove unneeded folders
			shutil.rmtree(args.pyframesPath)
			shutil.rmtree(args.pyaviPath)

if __name__ == '__main__':
    main()
