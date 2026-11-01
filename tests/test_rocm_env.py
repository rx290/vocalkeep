from core import rocm_env


def _fake_topology(monkeypatch, tmp_path, versions):
    paths = []
    for i, v in enumerate(versions):
        p = tmp_path / f"node{i}"
        p.write_text(f"cpu_cores_count 0\ngfx_target_version {v}\n")
        paths.append(str(p))
    monkeypatch.setattr(rocm_env.glob, "glob", lambda _: paths)


def test_rx6700xt_gets_rdna2_override(monkeypatch, tmp_path):
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    _fake_topology(monkeypatch, tmp_path, [0, 100301])     # CPU node, then gfx1031
    assert rocm_env.apply_rocm_override() == "10.3.0"


def test_natively_supported_card_left_alone(monkeypatch, tmp_path):
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    _fake_topology(monkeypatch, tmp_path, [0, 100300])     # gfx1030 ships in ROCm
    assert rocm_env.apply_rocm_override() is None


def test_user_value_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "9.9.9")
    _fake_topology(monkeypatch, tmp_path, [100301])
    assert rocm_env.apply_rocm_override() == "9.9.9"


def test_no_amd_gpu(monkeypatch, tmp_path):
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    _fake_topology(monkeypatch, tmp_path, [0])
    assert rocm_env.apply_rocm_override() is None
