from vidmeta.video.transcription import _format_transcript_with_speakers, TranscriptSegment


def test_transcript_with_face_labels_includes_legend():
    segments = [TranscriptSegment(start=0.0, end=1.0, text="Hello there")]
    speaker_turns = [(0.0, 1.0, "spk_0 / face_0")]
    text = _format_transcript_with_speakers(segments, speaker_turns)
    assert "Legend: Speaker N is the voice cluster" in text
    assert "Speaker 0 / Face 0" in text
