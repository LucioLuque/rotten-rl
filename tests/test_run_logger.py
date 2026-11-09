"""RunLogger: resuming a run truncates its CSVs, also after a machine crash."""
import os

from rotten_rl.run_logger import RunLogger


def test_resume_drops_rows_after_checkpoint_and_crash_debris(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    path = run / "eval.csv"
    path.write_bytes(b"episode,diff\n100,1.0\n200,2.0\n300,3.0\n30\x00\x00\x00\x00\n\x00\x00\x00")
    RunLogger(run_dir=str(run), config={}, resume_after=200)
    assert path.read_text().splitlines() == ["episode,diff", "100,1.0", "200,2.0"]
    assert os.path.isdir(run)
