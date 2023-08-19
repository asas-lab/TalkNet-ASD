import os, uuid, json, argparse, warnings

import pandas as pd

from pytube import YouTube, Search, Playlist
from pytube.exceptions import VideoUnavailable
import time
import cv2



warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "Download YouTube Videos")

parser.add_argument('--videoList',             type=str,  help='A CVS file containing a list of YouTube video URLs')
parser.add_argument('--videoFolder',           type=str, default="videos",  help='Path for inputs, tmps and outputs')

args = parser.parse_args()


def download(cvs_urls: str, out_dir: str):

    """
    Input:
        cvs_urls: A list of urls in the first col in CVS
        out_dir: A folder path to save the downloaded videos
    Return:
        download all videos in each podcast playlist
        json file conatining meta info about each video

    """
    podcasts = pd.read_csv(cvs_urls)
    podcasts_urls = podcasts.iloc[:, 1]

    # create directory folder for each podcast
    for i, url in enumerate(podcasts_urls):
        downloaded_podcast_path = os.path.join(out_dir, "podcast_" + str(i))
        if not os.path.exists(downloaded_podcast_path):
            os.makedirs(downloaded_podcast_path)

        # find all YouTube videos in podcast playlist
        yt = Playlist(url)



        for video_id, video_url in enumerate(yt):
            filename = str(video_id) + '.mp4'
            try:

                video = YouTube(video_url, use_oauth=True,
                                allow_oauth_cache=True)
                video.streams. \
                    filter(type='video', progressive=True, file_extension='mp4'). \
                    order_by('resolution'). \
                    desc(). \
                    first(). \
                    download(output_path=downloaded_podcast_path, filename=filename)



            except:
                print("Download Error!")

            meta_data = {}
            video_cv2 = cv2.VideoCapture(os.path.join(downloaded_podcast_path, filename))
            meta_data = {
                "video_id": filename.split('.')[0],
                "meta_info":
                    {
                        "video_title": video.title,
                        "duration": str(time.strftime("%H:%M:%S", time.gmtime(video.length))),
                        "channel_name": yt.title,
                        "creation_date": str(video.publish_date).split(' ')[0],
                        "ytb_id": video.video_id,
                        "video_size": {"height": video_cv2.get(cv2.CAP_PROP_FRAME_HEIGHT), "width": video_cv2.get(cv2.CAP_PROP_FRAME_WIDTH)},
                        "framerate": 25
                    }
            }

            metadate_dir = os.path.join(downloaded_podcast_path, "metadata")
            os.makedirs(metadate_dir, exist_ok=True)
            with open(os.path.join(metadate_dir, filename.split('.')[0] + '.json'), 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=4)


def main():
   
    args.pyaviPath = os.path.join(args.videoFolder)
    os.makedirs(args.pyaviPath, exist_ok = True) # The path for the input video, input audio, output video
    download(args.videoList, args.pyaviPath)


if __name__ == '__main__':
    main()
