"""Tests for the resumable-checkpoint helpers added to src/utils.py."""
import os
import json

from src.utils import load_checkpoint, save_checkpoint, clear_checkpoint


def test_load_checkpoint_missing_file_returns_empty(tmp_path):
    ckpt_file = str(tmp_path / "does_not_exist.json")
    data = load_checkpoint(ckpt_file)
    assert data == {'completed': [], 'results': []}


def test_save_then_load_checkpoint_roundtrip(tmp_path):
    ckpt_file = str(tmp_path / "ckpt.json")
    payload = {'completed': ['a.png', 'b.png'], 'results': [{'x': 1}]}
    save_checkpoint(ckpt_file, payload)

    loaded = load_checkpoint(ckpt_file)
    assert loaded == payload


def test_save_checkpoint_creates_parent_directory(tmp_path):
    ckpt_file = str(tmp_path / "nested" / "dir" / "ckpt.json")
    save_checkpoint(ckpt_file, {'completed': [], 'results': []})
    assert os.path.exists(ckpt_file)


def test_clear_checkpoint_removes_file(tmp_path):
    ckpt_file = str(tmp_path / "ckpt.json")
    save_checkpoint(ckpt_file, {'completed': ['x'], 'results': []})
    assert os.path.exists(ckpt_file)
    clear_checkpoint(ckpt_file)
    assert not os.path.exists(ckpt_file)


def test_clear_checkpoint_missing_file_does_not_raise(tmp_path):
    ckpt_file = str(tmp_path / "never_existed.json")
    clear_checkpoint(ckpt_file)  # should not raise


def test_save_checkpoint_is_atomic_no_tmp_file_left(tmp_path):
    ckpt_file = str(tmp_path / "ckpt.json")
    save_checkpoint(ckpt_file, {'completed': [], 'results': []})
    assert not os.path.exists(ckpt_file + '.tmp')
