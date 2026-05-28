from vidmeta.video.transcription import _align_speakers_to_faces


def test_align_speakers_to_faces_chooses_best_overlap():
    speaker_turns = [
        (0.0, 2.0, "spk_0"),
        (2.0, 4.0, "spk_1"),
    ]
    face_turns = [
        (0.5, 1.5, "face_0"),
        (2.2, 3.5, "face_1"),
    ]
    aligned = _align_speakers_to_faces(speaker_turns, face_turns)
    assert aligned == [
        (0.0, 2.0, "spk_0 / face_0"),
        (2.0, 4.0, "spk_1 / face_1"),
    ]
