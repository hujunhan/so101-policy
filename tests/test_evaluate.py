from so101_policy.evaluate import (
    append_row,
    final_rows,
    format_summary,
    paired,
    parse_trials,
    pending,
    placement_text,
    read_rows,
    summary,
)


def row(t, success="", failure="", void="", d="10.0"):
    return {
        "trial": str(t),
        "success": str(success),
        "failure": failure,
        "duration_s": d,
        "void": void,
        "notes": "",
    }


def test_log_roundtrip(tmp_path):
    p = tmp_path / "r" / "trials.csv"
    append_row(
        p, {"trial": 1, "success": 1, "failure": "", "duration_s": "12.0", "void": "", "notes": "a, b"}
    )
    append_row(
        p, {"trial": 2, "success": 0, "failure": "grasp_miss", "duration_s": "45.0", "void": "", "notes": ""}
    )
    rows = read_rows(p)
    assert [r["trial"] for r in rows] == ["1", "2"] and rows[0]["notes"] == "a, b"


def test_void_is_repeated_at_the_end_and_resume_skips_done():
    rows = [row(1, 1), row(2, void="camera dropout"), row(3, 0, "stuck")]
    assert pending([1, 2, 3, 4, 5], rows) == [4, 5, 2]
    rows.append(row(2, 1))
    assert pending([1, 2, 3, 4, 5], rows) == [4, 5]
    assert set(final_rows(rows)) == {1, 2, 3}


def test_summary_counts_and_interval():
    rows = [row(i, 1, d=str(10 + i)) for i in range(1, 25)] + [row(i, 0, "place_miss") for i in range(25, 31)]
    rows.append(row(30, void="bus error"))
    s = summary(rows)
    assert (s["n"], s["successes"], s["voids"]) == (30, 24, 1)
    assert s["failures"] == {"place_miss": 6}
    lo, hi = s["wilson95"]
    assert 0.6 < lo < 0.8 < hi < 0.95
    assert "24/30 = 80 %" in format_summary("act/base", s)


def test_paired_uses_common_trials():
    a = [row(1, 1), row(2, 1), row(3, 0, "stuck"), row(4, 1)]
    b = [row(1, 1), row(2, 0, "grasp_miss"), row(3, 1)]
    only_a, only_b, p, n = paired(a, b)
    assert (only_a, only_b, n) == (1, 1, 3) and p == 1.0


def test_parse_trials_and_placement():
    assert parse_trials("1-3,7") == [1, 2, 3, 7]
    assert len(parse_trials(None)) == 30
    st = {"u": 0.5, "v": 1.0, "yaw_deg": 30.0}
    region = {"origin": [200, -100], "u_axis": [0, 200], "v_axis": [150, 0]}
    assert "x 350 mm, y 0 mm" in placement_text(st, region)
    assert "region not fixed" in placement_text(st, None)
    assert "yaw" not in placement_text(st, region, yaw=False)


def test_protocol_region_matches_settings():
    from so101_policy.record import load_settings

    ev = load_settings()["eval"]
    st = {"u": 1.0, "v": 1.0, "yaw_deg": 0.0}
    assert "x 330 mm, y 40 mm" in placement_text(st, ev["region"], ev["use_yaw"])
