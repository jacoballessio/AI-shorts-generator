from generate_video_from_example import VideoGenerator

vg = VideoGenerator(use_pexels=False, use_elevenlabs=True)
print('eleven key', vg.elevenlabs_api_key)
print('trying sfx generation (should skip if no key)')
# provide a short duration to see that the parameter is forwarded
res = vg.generate_sfx_from_description('gunshot and explosion', 0, duration=1.5)
print('sfx result', res)
