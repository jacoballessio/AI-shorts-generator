import sys
import types

stub = types.ModuleType("asid_captioner")
stub.ASIDCaptioner = object
sys.modules["asid_captioner"] = stub

from generate_video_from_example import VideoGenerator


def main() -> None:
    generator = VideoGenerator(use_pexels=False, use_elevenlabs=False)
    clip = generator._create_caption_clip(
        text="urgency",
        color="#FFD54A",
        start=0.0,
        duration=0.5,
    )
    assert clip is not None
    assert generator._should_use_hunyuan_video("x" * 451) is True
    assert generator._should_use_hunyuan_video("x" * 450) is False
    print("smoke-ok")


if __name__ == "__main__":
    main()
