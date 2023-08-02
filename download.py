import os, uuid, json, argparse, warnings

import pandas as pd

from pytube import YouTube, Search, Playlist
from pytube.exceptions import VideoUnavailable



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
    video_urls = pd.read_csv(cvs_urls)
    urls = video_urls.iloc[:, 0]
	
    for i, url in enumerate(urls):
        downloaded_dir_path = os.path.join(out_dir, "video_" + str(i))
    if not os.path.exists(downloaded_dir_path):
        os.makedirs(downloaded_dir_path)

    meta_info = {}
    id = str(uuid.uuid4())
      
    for index, video_url in enumerate(urls):
        filename = str(index) + '.mp4'


          
        try:
            video = YouTube(video_url, use_oauth=False,
                    allow_oauth_cache=True)
            video.streams. \
                filter(type='video', progressive=True). \
                order_by('resolution'). \
                desc(). \
                first(). \
                download(output_path=downloaded_dir_path, filename=filename)

            meta_info[id] = {
                "video_title": video.title,
                "yt_url": video_url,
                "video_file_name": filename,
                "yt_channel_url": video.channel_url,
                "length": video.length
                    }
        except:
            print("Download Error!")


    #   with open("talking_head_info.json", "w") as outfile:
    #     json.dump(meta_info, outfile)

def main():
   
    args.pyaviPath = os.path.join(args.videoFolder)
    os.makedirs(args.pyaviPath, exist_ok = True) # The path for the input video, input audio, output video
    download(args.videoList, args.pyaviPath)


if __name__ == '__main__':
    main()
