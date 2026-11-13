from audio_captioner import AudioCaptioner
import time

print('[TEST] Creating AudioCaptioner...')
cap = AudioCaptioner()
print('[TEST] Instantiated AudioCaptioner')
start = time.time()
desc = cap.describe_video_audio('data/raw_videos/sENByrMHfyk.mp4')
print('[TEST] Finished audio captioning in', time.time() - start)
print('Description repr:', repr(desc))
