"""§10.8 Stage 1 GRADUATION (Фрай 10.06, направление B Шефа): durable-сайдкар →
ГРАФ-узел в cerebellum→motor контур. IN-MEMORY (спек остаётся сайдкарным →
рестарт = деградация в сайдкар), §3-gated (paralysis>0 → НЕМЕДЛЕННЫЙ revert),
rate-limit (Stage 1: ровно ОДНА graduation), мягкая стыковка (вес ребра φ⁻²).
Кандидат = только GC-KEEP-verified durable (max rise).

Тесты: флаг/гейты/лимит (dev-venv) + полный graduate/watch/revert цикл на real
neurocore (skip без core).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")

from utopia_client.local_compute import LocalColonyCompute  # noqa: E402


def _c():
    return LocalColonyCompute(device="cpu")


# ── флаг / дефолты ───────────────────────────────────────────────────────

def test_graduation_default_off():
    c = _c()
    assert c._tissue_graduation_enabled is False     # dormant дефолт
    assert c._tissue_grad_max == 89                  # Stage 2: лимит снят (Fib backstop)
    assert abs(c._TISSUE_GRAD_EDGE_WEIGHT - 0.382) < 1e-9   # φ⁻² мягкая стыковка


def test_set_tissue_graduation_toggle():
    c = _c()
    assert c.set_tissue_graduation(True) is True
    assert c.set_tissue_graduation(False) is False


# ── кандидаты: только GC-KEEP-verified durable ──────────────────────────

def test_no_candidate_without_gc_keep():
    # живой сайдкар есть, но GC ещё не подтвердил durable → graduation ждёт
    c = _c()
    org = types.SimpleNamespace(tissues={}, connections=[],
                                tissue_topology_genes=[])
    c.organisms["a"] = org
    c._grown_tissues["a"] = {"grown1": object()}
    assert c._graduate_tissue("a", org) is False
    assert "a" not in c._tissue_grad_state


def test_gc_keep_records_candidate_rise():
    c = _c()
    t = object()
    c._grown_tissues["a"] = {"grown1": t}
    c._tissue_grown_specs["a"] = [{"role": "grown1", "data_dim": 64, "n_embd": 21}]
    c.loss_ema["a"] = 0.05
    c._last_world_tick = 0
    c._maybe_start_tissue_gc("a")
    gc = c._tissue_gc_state["a"]
    gc["ticks"] = c._tissue_growth_dwell_ticks - 1
    c.loss_ema["a"] = 0.05 * (1 + c._growth_min_delta_frac + 0.01)   # durable
    c._resolve_tissue_gc("a", gc)
    assert "grown1" in c._tissue_gc_keep_rise["a"]
    assert c._tissue_gc_keep_rise["a"]["grown1"] > 0


def test_gc_prune_clears_candidate():
    c = _c()
    c._grown_tissues["a"] = {"grown1": object()}
    c._tissue_grown_specs["a"] = [{"role": "grown1", "data_dim": 64, "n_embd": 21}]
    c._tissue_gc_keep_rise["a"] = {"grown1": 0.02}    # KEEP прошлой эпохи
    c.loss_ema["a"] = 0.05
    c._last_world_tick = 0
    c._maybe_start_tissue_gc("a")
    gc = c._tissue_gc_state["a"]
    gc["ticks"] = c._tissue_growth_dwell_ticks - 1
    c.loss_ema["a"] = 0.05                            # вклад распался → прун
    c._resolve_tissue_gc("a", gc)
    assert "grown1" not in (c._tissue_gc_keep_rise.get("a") or {})


# ── Stage 1 лимит: одна graduation за сессию ────────────────────────────

def test_stage1_limit_blocks_second_graduation(monkeypatch):
    c = _c()
    org = types.SimpleNamespace(tissues={"cb": object()}, connections=[],
                                tissue_topology_genes=[])
    c.organisms["a"] = org
    c.predictor["a"] = object()
    c._tissue_growth_enabled = True
    c._tissue_graduation_enabled = True
    monkeypatch.setattr(c, "_cerebellum_tissue_id", lambda cid, o: "cb")
    monkeypatch.setattr(c, "_connections_saturated", lambda o: True)
    monkeypatch.setattr(c, "_intrinsic_plateaued", lambda cid: False)
    monkeypatch.setattr(c, "_maybe_start_tissue_gc", lambda cid: False)
    calls = []
    monkeypatch.setattr(c, "_graduate_tissue",
                        lambda cid, o: calls.append(cid) or True)
    monkeypatch.setattr(c, "_cumulative_grad_health", lambda cid, o: None)
    c._paralysis_window_n = 0
    c.biochem["a"] = types.SimpleNamespace(energy=1000.0)
    c._grad_health_streak["a"] = c._GRAD_HEALTH_TICKS    # health-гейт открыт
    c._tissue_grad_done = c._tissue_grad_max         # backstop исчерпан (89)
    c._tissue_growth_step("a")
    assert calls == []                               # graduation НЕ стартует
    c._tissue_grad_done = 0
    c._grad_health_streak["a"] = c._GRAD_HEALTH_TICKS
    c._tissue_growth_step("a")
    assert calls == ["a"]                            # лимит свободен → стартует


# ── полный цикл graduate/watch/revert (neurocore-gated) ─────────────────

def _grad_setup():
    """Адам-минимум: cerebellum в графе + durable сайдкар grown1."""
    pytest.importorskip("core.tissue")
    c = _c()
    cb = c._make_higher_tissue("cerebellum", data_dim=64)
    if cb is None:
        pytest.skip("cerebellum tissue build failed (neurocore?)")
    t = c._make_higher_tissue("grown1", data_dim=64, n_embd=21)
    org = types.SimpleNamespace(
        tissues={getattr(cb, "tissue_id", "cb"): cb}, connections=[],
        tissue_topology_genes=[])
    c.organisms["a"] = org
    c.predictor["a"] = object()
    c.loss_ema["a"] = 0.05
    c._last_world_tick = 1000
    c._grown_tissues["a"] = {"grown1": t}
    c._tissue_gc_keep_rise["a"] = {"grown1": 0.02}
    c.biochem["a"] = types.SimpleNamespace(energy=1000.0)
    return c, org, t


def test_graduate_inserts_node_and_soft_edge():
    c, org, t = _grad_setup()
    assert c._graduate_tissue("a", org) is True
    tid = getattr(t, "tissue_id")
    # узел в графе, сайдкар ушёл из прямого входа предиктора
    assert org.tissues.get(tid) is t
    assert "grown1" not in (c._grown_tissues.get("a") or {})
    assert c._tissue_graduated["a"]["grown1"] is t
    # ген включён, вес φ⁻² (мягкая стыковка, НЕ 1.0)
    g = next(g for g in org.tissue_topology_genes
             if g.source_role == "grown1" and g.target_role == "cerebellum")
    assert g.enabled and abs(g.weight - 0.382) < 1e-9
    # overlay провёл ребро grown1→cerebellum в org.connections
    assert any(cn.source_tissue_id == tid for cn in org.connections)
    # watch-окно открыто
    st = c._tissue_grad_state["a"]
    assert st["role"] == "grown1" and st["par_before"] == 0


def test_graduate_picks_max_rise():
    c, org, t = _grad_setup()
    t2 = c._make_higher_tissue("grown2", data_dim=64, n_embd=21)
    c._grown_tissues["a"]["grown2"] = t2
    c._tissue_gc_keep_rise["a"] = {"grown1": 0.01, "grown2": 0.03}
    assert c._graduate_tissue("a", org) is True
    assert c._tissue_grad_state["a"]["role"] == "grown2"   # max rise


def test_watch_paralysis_immediate_revert():
    c, org, t = _grad_setup()
    c._graduate_tissue("a", org)
    st = c._tissue_grad_state["a"]
    c._paralysis_window_n = 1                        # §3-сигнал
    c._tissue_graduation_watch("a", org, st)         # НЕ ждёт конца окна
    tid = getattr(t, "tissue_id")
    assert tid not in org.tissues                    # узел из графа
    assert c._grown_tissues["a"]["grown1"] is t      # сайдкар назад, ТОТ ЖЕ объект
    assert all(not g.enabled for g in org.tissue_topology_genes
               if g.source_role == "grown1")
    assert not any(cn.source_tissue_id == tid for cn in org.connections)
    assert c._tissue_grad_reverted == 1
    assert "a" not in c._tissue_grad_state
    assert c._tissue_grad_done == 0


def test_watch_energy_collapse_revert():
    c, org, t = _grad_setup()
    c._graduate_tissue("a", org)
    st = c._tissue_grad_state["a"]
    c.biochem["a"].energy = 500.0                    # < 0.618 × 1000
    for _ in range(c._GRAD_COLLAPSE_WIN):            # window-based (анти-churn)
        if "a" not in c._tissue_grad_state:
            break
        c._tissue_graduation_watch("a", org, st)
    assert c._tissue_grad_reverted == 1
    assert c._grown_tissues["a"]["grown1"] is t


def test_watch_slow_energy_bleed_reverts_at_window_end():
    # Фрай 10.06: первый §3 был МЕДЛЕННЫЙ bleed (форейдж деградировал
    # постепенно) — мгновенные детекторы (paralysis, <φ⁻¹) его прозевают.
    # Кумулятивный тренд: avg(2-я половина) < avg(1-я)·(1−φ⁻⁵) → revert.
    c, org, t = _grad_setup()
    c._graduate_tissue("a", org)
    st = c._tissue_grad_state["a"]
    dwell = c._tissue_growth_dwell_ticks
    for i in range(dwell):
        # линейный bleed 1000→800 (−20%): мгновенный порог 618 НЕ задет
        c.biochem["a"].energy = 1000.0 - 200.0 * (i + 1) / dwell
        c._tissue_graduation_watch("a", org, st)
        if "a" not in c._tissue_grad_state:
            break
    assert c._tissue_grad_reverted == 1              # bleed пойман трендом
    assert c._tissue_grad_done == 0
    assert c._grown_tissues["a"]["grown1"] is t      # деградация в сайдкар


def test_watch_healthy_oscillation_graduates():
    # здоровая осцилляция energy (еда/трата) вокруг уровня — НЕ bleed, OK.
    c, org, t = _grad_setup()
    c._graduate_tissue("a", org)
    st = c._tissue_grad_state["a"]
    dwell = c._tissue_growth_dwell_ticks
    for i in range(dwell):
        c.biochem["a"].energy = 950.0 + (50.0 if i % 2 else -50.0)
        c._tissue_graduation_watch("a", org, st)
    assert c._tissue_grad_done == 1                  # окно чисто
    assert c._tissue_grad_reverted == 0
    assert org.tissues.get(getattr(t, "tissue_id")) is t


def test_watch_clean_window_graduates():
    c, org, t = _grad_setup()
    c._graduate_tissue("a", org)
    st = c._tissue_grad_state["a"]
    st["ticks"] = c._tissue_growth_dwell_ticks - 1
    c._tissue_graduation_watch("a", org, st)         # окно чисто
    tid = getattr(t, "tissue_id")
    assert c._tissue_grad_done == 1
    assert "a" not in c._tissue_grad_state
    assert org.tissues.get(tid) is t                 # узел ЖИВЁТ в графе
    assert c._tissue_graduated["a"]["grown1"] is t


def test_regraduation_reuses_gene_no_duplicate():
    c, org, t = _grad_setup()
    c._graduate_tissue("a", org)
    c._revert_graduation("a", org, c._tissue_grad_state["a"], reason="test")
    c._last_world_tick += c._tissue_gc_epoch_interval   # cooldown (2) истёк
    c._graduate_tissue("a", org)                     # повторная попытка
    genes = [g for g in org.tissue_topology_genes
             if g.source_role == "grown1" and g.target_role == "cerebellum"]
    assert len(genes) == 1 and genes[0].enabled      # ген ОДИН (нет дубля сигнала)


# ── kill-switches ────────────────────────────────────────────────────────

def test_graduation_kill_switch_reverts_node():
    c, org, t = _grad_setup()
    c._tissue_graduation_enabled = True
    c._graduate_tissue("a", org)
    st = c._tissue_grad_state["a"]
    st["ticks"] = c._tissue_growth_dwell_ticks - 1
    c._tissue_graduation_watch("a", org, st)         # GRADUATE-OK
    c.set_tissue_graduation(False)                   # kill-switch
    assert getattr(t, "tissue_id") not in org.tissues
    assert c._grown_tissues["a"]["grown1"] is t      # деградация в сайдкар


def test_tissue_growth_kill_switch_reverts_graduated_too():
    c, org, t = _grad_setup()
    c._tissue_growth_enabled = True
    c._graduate_tissue("a", org)
    c.set_tissue_growth(False)                       # ПОЛНЫЙ РЕВЕРТ
    assert getattr(t, "tissue_id") not in org.tissues   # узла нет в графе
    assert not c._grown_tissues.get("a")             # и сайдкаров нет
    assert not c._tissue_graduated.get("a")
    assert not c._tissue_gc_keep_rise.get("a")


# ── in-memory персист-семантика: рестарт = деградация в сайдкар ─────────

def test_save_keeps_graduated_weights_and_sidecar_spec():
    c, org, t = _grad_setup()
    org.generation = 0
    c._tissue_grown_specs["a"] = [{"role": "grown1", "data_dim": 64, "n_embd": 21}]
    c._graduate_tissue("a", org)
    payload = c.save_state("a")
    # спек ОСТАЛСЯ сайдкарным → restore пересоздаст сайдкар (деградация)
    assert payload["growth_loop"]["grown_tissues"] == [
        {"role": "grown1", "data_dim": 64, "n_embd": 21}]
    # веса graduated-узла в grown_weights → обучение НЕ теряется на рестарте
    assert "grown1" in payload.get("grown_weights", {})


# ── STAGE 2 PERSIST (Фрай go 10.06): graduated переживает рестарт В ГРАФЕ ──

def _graduate_ok(c, org):
    """Довести graduation до GRADUATE-OK (watch завершён чисто)."""
    st = c._tissue_grad_state["a"]
    st["ticks"] = c._tissue_growth_dwell_ticks - 1
    c._tissue_graduation_watch("a", org, st)
    assert c._tissue_grad_done >= 1


def test_save_marks_graduated_separately():
    c, org, t = _grad_setup()
    org.generation = 0
    c._tissue_grown_specs["a"] = [{"role": "grown1", "data_dim": 64, "n_embd": 21}]
    c._graduate_tissue("a", org)
    _graduate_ok(c, org)
    payload = c.save_state("a")
    gl = payload["growth_loop"]
    assert gl["graduated_tissues"] == [
        {"role": "grown1", "data_dim": 64, "n_embd": 21}]
    # backward-compat: роль ОСТАЁТСЯ в grown_tissues (старый клиент → сайдкар)
    assert gl["grown_tissues"] == [
        {"role": "grown1", "data_dim": 64, "n_embd": 21}]
    assert "grown1" in payload["grown_weights"]


def _restore_target():
    """Свежий компьют + орг-скелет с cerebellum (как после add_creature)."""
    c = _c()
    cb = c._make_higher_tissue("cerebellum", data_dim=64)
    org = types.SimpleNamespace(
        tissues={getattr(cb, "tissue_id", "cb"): cb}, connections=[],
        tissue_topology_genes=[], generation=0)
    c.organisms["a"] = org
    return c, org


def test_restore_recreates_graduated_in_graph():
    src, org_s, t = _grad_setup()
    org_s.generation = 0
    src._tissue_grown_specs["a"] = [{"role": "grown1", "data_dim": 64, "n_embd": 21}]
    src._graduate_tissue("a", org_s)
    _graduate_ok(src, org_s)
    payload = src.save_state("a")

    dst, org_d = _restore_target()
    dst.restore_persisted_state("a", payload)
    # узел В ГРАФЕ (не сайдкаром), ребро проведено overlay'ем
    roles = {getattr(tt, "role", None) for tt in org_d.tissues.values()}
    assert "grown1" in roles
    assert "grown1" in dst._tissue_graduated.get("a", {})
    assert "grown1" not in (dst._grown_tissues.get("a") or {})   # НЕ двойник
    tid = next(k for k, tt in org_d.tissues.items()
               if getattr(tt, "role", None) == "grown1")
    assert any(cn.source_tissue_id == tid for cn in org_d.connections)
    # счётчик Stage-1 лимита восстановлен → новых graduations не будет
    assert dst._tissue_grad_done == 1


def test_restore_without_gene_degrades_to_sidecar():
    # ГАРД: graduated-спек есть, а ген пропал → узел НЕ вставляем (orphan
    # стал бы мотор-выходом) → деградация в сайдкар (Stage 1 семантика).
    src, org_s, t = _grad_setup()
    org_s.generation = 0
    src._tissue_grown_specs["a"] = [{"role": "grown1", "data_dim": 64, "n_embd": 21}]
    src._graduate_tissue("a", org_s)
    _graduate_ok(src, org_s)
    payload = src.save_state("a")
    payload["tissue_topology_genes"] = []            # ген «пропал»

    dst, org_d = _restore_target()
    dst.restore_persisted_state("a", payload)
    roles = {getattr(tt, "role", None) for tt in org_d.tissues.values()}
    assert "grown1" not in roles                     # в графе НЕТ
    assert "grown1" in (dst._grown_tissues.get("a") or {})   # сайдкар-fallback
    assert not dst._tissue_graduated.get("a")


def test_restore_old_payload_without_stage2_degrades():
    # payload Stage 1 (без graduated_tissues) → как раньше: сайдкар.
    src, org_s, t = _grad_setup()
    org_s.generation = 0
    src._tissue_grown_specs["a"] = [{"role": "grown1", "data_dim": 64, "n_embd": 21}]
    src._graduate_tissue("a", org_s)
    _graduate_ok(src, org_s)
    payload = src.save_state("a")
    payload["growth_loop"].pop("graduated_tissues", None)

    dst, org_d = _restore_target()
    dst.restore_persisted_state("a", payload)
    assert "grown1" in (dst._grown_tissues.get("a") or {})
    assert not dst._tissue_graduated.get("a")


def test_restored_graduated_killswitch_reverts():
    src, org_s, t = _grad_setup()
    org_s.generation = 0
    src._tissue_grown_specs["a"] = [{"role": "grown1", "data_dim": 64, "n_embd": 21}]
    src._graduate_tissue("a", org_s)
    _graduate_ok(src, org_s)
    payload = src.save_state("a")

    dst, org_d = _restore_target()
    dst.restore_persisted_state("a", payload)
    dst.set_tissue_graduation(False)                 # kill-switch после рестарта
    roles = {getattr(tt, "role", None) for tt in org_d.tissues.values()}
    assert "grown1" not in roles
    assert "grown1" in (dst._grown_tissues.get("a") or {})    # обратно в сайдкар


def test_midwatch_save_degrades_to_sidecar():
    # сейв ПОСРЕДИ watch-окна: §3-клиренса ещё нет → консервативно НЕ
    # помечаем graduated → restore даст сайдкар, graduation пройдёт заново.
    src, org_s, t = _grad_setup()
    org_s.generation = 0
    src._tissue_grown_specs["a"] = [{"role": "grown1", "data_dim": 64, "n_embd": 21}]
    src._graduate_tissue("a", org_s)                 # watch in-flight
    payload = src.save_state("a")
    assert payload["growth_loop"]["graduated_tissues"] == []
    dst, org_d = _restore_target()
    dst.restore_persisted_state("a", payload)
    assert "grown1" in (dst._grown_tissues.get("a") or {})
    assert not dst._tissue_graduated.get("a")


def test_graduation_metrics_for_ui():
    c = _c()
    c._tissue_graduated["a"] = {"grown1": object()}
    c._tissue_grad_done = 1
    c._tissue_grad_reverted = 2
    m = c._tissue_growth_metrics("a")
    assert m["tissue_graduated_live"] == 1
    assert m["tissue_grad_watch"] == 0
    assert m["tissue_grad_done"] == 1 and m["tissue_grad_reverted"] == 2
    assert m["tissue_graduation_enabled"] is False
    # graduated НЕ входит в tissue_grown_live (узел в графе ≠ сайдкар)
    assert m["tissue_grown_live"] == 0
    # агрегат
    assert c._tissue_growth_metrics()["tissue_graduated_live"] == 1


# ── АНТИ-CHURN GUARD (Фрай 10.06, инцидент grown151) ────────────────────

def test_revert_sets_role_cooldown_and_streak():
    c, org, t = _grad_setup()
    c._graduate_tissue("a", org)
    c._last_world_tick = 2000
    c._revert_graduation("a", org, c._tissue_grad_state["a"], reason="energy-collapse")
    assert c._grad_rejected["a"]["grown1"] == 2000   # (2) роль в cooldown
    assert c._grad_revert_streak == 1
    assert c._grad_health_streak.get("a", 0) == 0    # (A) recovery заново
    # cooldown: та же роль НЕ выпускается сразу
    assert c._graduate_tissue("a", org) is False


def test_revert_streak_halts_graduation():
    c = _c()
    org = types.SimpleNamespace(tissue_topology_genes=[], connections=[], tissues={})
    c.organisms["a"] = org
    c._degraduate_node = lambda *a, **k: None
    for i in range(c._GRAD_REVERT_HALT):
        c._tissue_grad_state["a"] = {"role": f"grown{i}"}
        c._revert_graduation("a", org, c._tissue_grad_state["a"], reason="x")
    assert c._grad_halted is True                     # (3) HALT


def test_reflip_clears_halt():
    c = _c()
    c._grad_halted = True
    c._grad_revert_streak = 3
    c.set_tissue_graduation(True)
    assert c._grad_halted is False and c._grad_revert_streak == 0


def test_killswitch_revert_no_streak():
    # kill-switch не считается churn-revert'ом (не копит streak/cooldown)
    c, org, t = _grad_setup()
    c._graduate_tissue("a", org)
    c._revert_graduation("a", org, c._tissue_grad_state["a"], reason="kill-switch")
    assert c._grad_revert_streak == 0
    assert not c._grad_rejected.get("a")


def test_collapse_detector_is_window_based():
    # одиночный погодный провал НЕ роняет watch; устойчиво низкое окно — роняет
    c, org, t = _grad_setup()
    c._graduate_tissue("a", org)
    st = c._tissue_grad_state["a"]
    c.biochem["a"].energy = 100.0                    # глубокий провал 1 тик
    c._tissue_graduation_watch("a", org, st)
    assert "a" in c._tissue_grad_state               # выжил (rolling-mean)
    for _ in range(c._GRAD_COLLAPSE_WIN):
        if "a" not in c._tissue_grad_state:
            break
        c._tissue_graduation_watch("a", org, st)     # 13 тиков низко
    assert "a" not in c._tissue_grad_state           # collapse пойман
    assert c._tissue_grad_reverted == 1


def test_health_streak_gates_graduation(monkeypatch):
    c = _c()
    org = types.SimpleNamespace(tissues={"cb": object()}, connections=[],
                                tissue_topology_genes=[])
    c.organisms["a"] = org
    c.predictor["a"] = object()
    c._tissue_growth_enabled = True
    c._tissue_graduation_enabled = True
    monkeypatch.setattr(c, "_cerebellum_tissue_id", lambda cid, o: "cb")
    monkeypatch.setattr(c, "_connections_saturated", lambda o: True)
    monkeypatch.setattr(c, "_intrinsic_plateaued", lambda cid: False)
    monkeypatch.setattr(c, "_maybe_start_tissue_gc", lambda cid: False)
    calls = []
    monkeypatch.setattr(c, "_graduate_tissue",
                        lambda cid, o: calls.append(cid) or True)
    c.biochem["a"] = types.SimpleNamespace(energy=1000.0)
    c._paralysis_window_n = 0
    # здоров, но streak ещё не набран → выпуска нет
    for _ in range(c._GRAD_HEALTH_TICKS - 1):
        c._tissue_growth_step("a")
    assert calls == []
    c._tissue_growth_step("a")                       # 89-й тик подряд
    assert calls == ["a"]                            # гейт открылся
    # провал энергии сбрасывает streak
    c.biochem["a"].energy = 300.0
    c._tissue_growth_step("a")
    assert c._grad_health_streak["a"] == 0


# ── эскалация cooldown (Фрай: 2×revert = проверенно-вредная) ────────────

def test_cooldown_escalates_with_revert_count():
    c, org, t = _grad_setup()
    c._graduate_tissue("a", org)
    c._last_world_tick = 10000
    c._revert_graduation("a", org, c._tissue_grad_state["a"], reason="x")
    assert c._grad_revert_count["a"]["grown1"] == 1
    # 1й revert: cooldown 1×epoch — после него снова кандидат
    c._last_world_tick = 10000 + c._tissue_gc_epoch_interval
    assert c._graduate_tissue("a", org) is True
    c._revert_graduation("a", org, c._tissue_grad_state["a"], reason="x")
    assert c._grad_revert_count["a"]["grown1"] == 2
    # 2й revert: cooldown 2×epoch — 1×epoch уже НЕ хватает
    c._last_world_tick += c._tissue_gc_epoch_interval
    assert c._graduate_tissue("a", org) is False
    c._last_world_tick += c._tissue_gc_epoch_interval      # итого 2×
    assert c._graduate_tissue("a", org) is True


def test_revert_count_persists():
    src = _c()
    src.organisms["a"] = types.SimpleNamespace(generation=0)
    src._grad_revert_count["a"] = {"grown151": 2}
    payload = src.save_state("a")
    assert payload["growth_loop"]["grad_revert_count"] == {"grown151": 2}
    dst = _c()
    dst.organisms["a"] = types.SimpleNamespace(generation=0)
    dst.restore_persisted_state("a", payload)
    assert dst._grad_revert_count["a"] == {"grown151": 2}


# ── grad_max снят + CUMULATIVE-HEALTH-MONITOR (Фрай 11.06) ───────────────

def test_grad_max_lifted():
    c = _c()
    assert c._tissue_grad_max == 89          # Fib backstop, не тесный 1


def test_cumulative_monitor_inactive_at_zero():
    c = _c()
    org = types.SimpleNamespace()
    c.organisms["a"] = org
    c._tissue_graduated["a"] = {}              # 0 graduated → нечему дрейфовать
    c.biochem["a"] = types.SimpleNamespace(energy=100.0)
    c._grad_collective_paused = True
    c._cumulative_grad_health("a", org)
    assert c._grad_collective_paused is False  # 0 → монитор неактивен


def test_cumulative_monitor_active_at_one(monkeypatch):
    # Фрай 11.06: монитор теперь активен при ≥1 (инцидент 52%-паралич — одиночный
    # узел дренил без страховки). 1 graduated + дрейф → SHED (auto-защита).
    c = _c()
    org = types.SimpleNamespace()
    c.organisms["a"] = org
    shed = []
    c._degraduate_node = lambda cid, o, role, reason="": shed.append(role)
    c._tissue_graduated["a"] = {"grown1": object()}   # ОДИН узел
    c._grad_value["a"] = {"grown1": 0.05}
    c.biochem["a"] = types.SimpleNamespace(energy=50.0)   # < _GRAD_SHED_ENERGY → drift
    c._paralysis_window_n = 1
    c._cumulative_grad_health("a", org)
    assert c._grad_collective_paused is True   # активен при N=1 → пауза
    assert shed == ["grown1"]                  # одиночка shed'ится (recoverable)


def test_cumulative_monitor_sheds_lowest_value_on_drift():
    c = _c()
    org = types.SimpleNamespace(tissue_topology_genes=[], connections=[], tissues={})
    c.organisms["a"] = org
    c._degraduate_node = lambda cid, o, role, reason="": c._tissue_graduated["a"].pop(role, None)
    c._tissue_graduated["a"] = {"grown1": object(), "grown2": object(), "grown3": object()}
    c._grad_value["a"] = {"grown1": 0.05, "grown2": 10.5, "grown3": 0.01}  # grown3 ниже
    c.biochem["a"] = types.SimpleNamespace(energy=50.0)   # просадка
    c._paralysis_window_n = 3                             # §3-дрейф
    c._cumulative_grad_health("a", org)
    assert c._grad_collective_paused is True              # пауза
    assert "grown3" not in c._tissue_graduated["a"]       # shed наименее-ценного
    assert "grown2" in c._tissue_graduated["a"]           # ценный (beh-KEEP) жив


def test_cumulative_monitor_resumes_on_recovery():
    c = _c()
    org = types.SimpleNamespace()
    c.organisms["a"] = org
    c._tissue_graduated["a"] = {"grown1": object(), "grown2": object()}
    c._grad_collective_paused = True
    c.biochem["a"] = types.SimpleNamespace(energy=1000.0)  # здоров
    c._paralysis_window_n = 0
    # EWMA должна подняться к recovery-порогу — несколько тиков
    for _ in range(60):
        c._cumulative_grad_health("a", org)
    assert c._grad_collective_paused is False             # возобновлён


def test_graduation_gated_by_collective_pause(monkeypatch):
    c = _c()
    org = types.SimpleNamespace(tissues={"cb": object()}, connections=[],
                                tissue_topology_genes=[])
    c.organisms["a"] = org
    c.predictor["a"] = object()
    c._tissue_growth_enabled = True
    c._tissue_graduation_enabled = True
    monkeypatch.setattr(c, "_cerebellum_tissue_id", lambda cid, o: "cb")
    monkeypatch.setattr(c, "_connections_saturated", lambda o: True)
    monkeypatch.setattr(c, "_cumulative_grad_health", lambda cid, o: None)
    monkeypatch.setattr(c, "_intrinsic_plateaued", lambda cid: False)
    monkeypatch.setattr(c, "_maybe_start_tissue_gc", lambda cid: False)
    calls = []
    monkeypatch.setattr(c, "_graduate_tissue", lambda cid, o: calls.append(cid) or True)
    c.biochem["a"] = types.SimpleNamespace(energy=1000.0)
    c._paralysis_window_n = 0
    c._grad_health_streak["a"] = c._GRAD_HEALTH_TICKS
    c._grad_collective_paused = True          # пауза от collective-drift
    c._tissue_growth_step("a")
    assert calls == []                         # graduation НЕ стартует под паузой
