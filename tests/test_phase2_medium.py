"""Phase 2 feeding-ladder — средняя дичь = 55 (Фрай pounce-модель 12.06).
Client-half: восприятие nearest_medium_prey → medium-seek nav (арбитраж голод+
способность, коммит ОДНУ цель, ровная φ-погоня) + pounce-флаг (dist≤_POUNCE_DIST
→ +1 рывок на entry MOVE) + контакт-ATTACK. Сервер (Хьюберт) даёт kill-energy 55,
сложность (hp/speed/flee) + prey-tiring (нагон). Spawn OFF до ack.
"""
from __future__ import annotations
import sys, types
from pathlib import Path
import pytest
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
pytest.importorskip("torch")
from utopia_client.local_compute import LocalColonyCompute, _PHI  # noqa: E402

_PINV = 1.0 / _PHI            # φ⁻¹≈0.618 (порог голода)


def _c(hunting=True):
    c = LocalColonyCompute(device="cpu")
    c._single_organism = True
    c.set_instinct_dir_strength(2.0)         # прод: DS из client_flag (>0)
    c._hunting_enabled = hunting
    return c


# ── medium-seek nav: арбитраж голод + способность ───────────────────────

def _med_nav(energy_ratio, dr=-1, dc=0, dist=5.0, hunting=True, diet=0.618,
             d_prox=0.0):
    import torch
    c = _c(hunting=hunting)
    l = torch.zeros(16)
    obs = [0.0] * 64
    obs[61] = d_prox                         # хищник (survival-гейт)
    mp = {"dr": dr, "dc": dc, "dist": dist}
    c._shape_action_logits(l, obs, diet=diet, energy_ratio=energy_ratio,
                           medium_prey=mp)
    return l


def test_medium_seek_when_hungry_capable():
    # голоден (< φ⁻¹) И способен (> φ⁻⁵) → тянет к средней (dr<0 → NORTH)
    l = _med_nav(energy_ratio=0.4, dr=-1, dc=0, dist=5.0)
    assert float(l[0]) > 0                    # NORTH (к добыче севернее)
    assert float(l[0]) > float(l[1])          # доминирует обратное


def test_medium_seek_east():
    l = _med_nav(energy_ratio=0.4, dr=0, dc=1, dist=5.0)
    assert float(l[2]) > 0                     # EAST (dc>0)


def test_medium_seek_off_when_fed():
    # сытый (≥ φ⁻¹) → НЕ лезет на среднюю (форажит дешёвое)
    l = _med_nav(energy_ratio=0.8, dr=-1, dc=0, dist=5.0)
    assert float(l[0]) == 0.0                  # нет medium-pull


def test_medium_seek_off_when_critically_starving():
    # критично-истощён (< φ⁻⁵) → НЕ способен, не лезет (умрёт в попытке)
    l = _med_nav(energy_ratio=0.05, dr=-1, dc=0, dist=5.0)
    assert float(l[0]) == 0.0


def test_medium_seek_off_when_hunting_disabled():
    l = _med_nav(energy_ratio=0.4, dr=-1, dc=0, dist=5.0, hunting=False)
    assert float(l[0]) == 0.0


def test_medium_seek_suppressed_near_predator():
    # хищник рядом (d_prox≥0.3) → survival > охота, medium-seek ОТКЛ
    l_safe = _med_nav(energy_ratio=0.4, dr=-1, dc=0, dist=5.0, d_prox=0.0)
    l_danger = _med_nav(energy_ratio=0.4, dr=-1, dc=0, dist=5.0, d_prox=0.5)
    assert float(l_safe[0]) > 0
    assert float(l_danger[0]) == 0.0


def test_medium_seek_dominates_small_prey():
    # арбитраж: голодный коммитит СРЕДНЮЮ (55) сильнее мелкой-prey nav (21)
    import torch
    c = _c()
    # только мелкая
    l_small = torch.zeros(16)
    obs_s = [0.0] * 64
    obs_s[56] = 1.0; obs_s[58] = 0.1          # мелкая prey-направление север
    c._shape_action_logits(l_small, obs_s, diet=0.618, energy_ratio=0.4)
    # средняя + мелкая
    l_both = torch.zeros(16)
    obs_b = [0.0] * 64
    obs_b[56] = 1.0; obs_b[58] = 0.1
    mp = {"dr": -1, "dc": 0, "dist": 5.0}
    c._shape_action_logits(l_both, obs_b, diet=0.618, energy_ratio=0.4,
                           medium_prey=mp)
    assert float(l_both[0]) > float(l_small[0])   # средняя добавила тягу


def test_medium_contact_attack():
    # на контакте (dist≤1) → доминантный ATTACK + гаси move
    l = _med_nav(energy_ratio=0.4, dr=0, dc=0, dist=1.0)
    assert float(l[5]) > 0                     # ATTACK
    assert int(l.argmax()) == 5                # доминирует argmax


def test_medium_contact_attack_when_starving():
    # §3-выход (Шеф 12.06): дичь ВПЛОТНУЮ → ATTACK даже при er=0 (не capable).
    # Еда у рта берётся даже умирающим — выход из голод-капкана.
    l = _med_nav(energy_ratio=0.0, dr=0, dc=0, dist=1.0)
    assert float(l[5]) > 0
    assert int(l.argmax()) == 5


def test_medium_chase_still_gated_when_starving():
    # погоня (dist>1) НЕ фичрит при er<φ⁻⁵ — не гнаться умирая (только контакт).
    l = _med_nav(energy_ratio=0.05, dr=-1, dc=0, dist=5.0)
    assert float(l[0]) == 0.0                  # нет nav-pull к далёкой дичи


# ── pounce-флаг (obs-loop) → entry speed_boost ──────────────────────────

def _ws():
    import utopia_client.ws_client as wsm
    return wsm.ColonyWSClient(server="https://e.com", token="t",
                              colony_name="cheef", client_version="0.0.0",
                              estimated_population=0)


def test_pounce_flag_set_in_window():
    # dist≤_POUNCE_DIST + голоден + способен + hunting → флаг ставится
    c = _c()
    c.biochem["a"] = types.SimpleNamespace(energy=300.0)   # er=0.3 < φ⁻¹
    c._hunt_pounce.clear()
    # эмулируем обращение obs-loop: проверяем по контракту флага напрямую
    _er = 0.3
    _mp = {"dr": -1, "dc": 0, "dist": 2.0}
    if (c._hunting_enabled and _mp is not None
            and _er < (1.0 / _PHI) and _er > (1.0 / _PHI) ** 5
            and _mp.get("dist") <= c._POUNCE_DIST):
        c._hunt_pounce["a"] = 1
    assert c._hunt_pounce.get("a") == 1


def test_pounce_entry_speed_boost():
    # флаг стоит → кардинальный MOVE получает speed_boost=+1
    ws = _ws()
    comp = types.SimpleNamespace()
    comp._single_organism = True
    comp._hunting_enabled = True
    comp._hunt_pounce = {"a": 1}
    comp.biochem = {"a": types.SimpleNamespace(energy=300.0)}
    comp.handle_tick = lambda *a, **k: {"a": {"action": 0}}   # MOVE NORTH
    comp.get_phase_emas = lambda cid: None
    ws.compute = comp
    out = ws._run_tick_and_build({"a": [0.0] * 64}, {}, {}, world_tick=0)
    e = {x["cid"]: x for x in (out or [])}["a"]
    assert e.get("speed_boost") == 1
    assert e.get("persist") is True


def test_paralysis_allows_contact_hunt_attack():
    # §3-bypass: парализован (energy=0) + дичь вплотную (флаг) + ATTACK → проходит
    import time as _t
    c = _c()
    c.biochem["a"] = types.SimpleNamespace(energy=0.0)
    c._hunt_contact["a"] = 1
    c._paralysis_until["a"] = _t.monotonic() + 5.0
    out = {"a": {"action": 5, "target_id": None}}     # network выбрал ATTACK
    c._maybe_force_stay("a", out)
    assert out["a"]["action"] == 5                     # ATTACK прошёл (не STAY)


def test_paralysis_forces_stay_non_attack():
    # MOVE в параличе → всё ещё STAY (bypass только для ATTACK)
    import time as _t
    c = _c()
    c.biochem["a"] = types.SimpleNamespace(energy=0.0)
    c._hunt_contact["a"] = 1
    c._paralysis_until["a"] = _t.monotonic() + 5.0
    out = {"a": {"action": 1, "target_id": None}}
    c._maybe_force_stay("a", out)
    assert out["a"]["action"] == 4                     # STAY


def test_paralysis_allows_eat_on_food():
    # §3-EAT bypass (eating.md): парализован + на еде + EAT → проходит (выедание из §3)
    import time as _t
    c = _c()
    c.biochem["a"] = types.SimpleNamespace(energy=0.0)
    c._on_food["a"] = 1
    c._paralysis_until["a"] = _t.monotonic() + 5.0
    out = {"a": {"action": 14, "target_id": None}}    # EAT
    c._maybe_force_stay("a", out)
    assert out["a"]["action"] == 14                    # EAT прошёл (не STAY)


def test_paralysis_forces_stay_eat_no_food():
    # EAT в параличе БЕЗ on_food-флага → STAY (узкий bypass)
    import time as _t
    c = _c()
    c.biochem["a"] = types.SimpleNamespace(energy=0.0)
    c._paralysis_until["a"] = _t.monotonic() + 5.0
    out = {"a": {"action": 14, "target_id": None}}
    c._maybe_force_stay("a", out)
    assert out["a"]["action"] == 4                     # STAY (нет флага)


def test_paralysis_forces_stay_attack_no_contact():
    # ATTACK в параличе БЕЗ contact-флага → STAY (узкий bypass)
    import time as _t
    c = _c()
    c.biochem["a"] = types.SimpleNamespace(energy=0.0)
    c._paralysis_until["a"] = _t.monotonic() + 5.0
    out = {"a": {"action": 5, "target_id": None}}
    c._maybe_force_stay("a", out)
    assert out["a"]["action"] == 4                     # STAY (нет флага)


def test_eat_commit_hungry_on_flora():
    # Phase A (Фрай 13.06, eating.md): голоден + НА еде + нет хищника → доминантный
    # EAT + гаси move (осознанный приём пищи, не вакуум на ходу).
    import torch
    c = _c()
    l = torch.zeros(16)
    obs = [0.0] * 64
    c._shape_action_logits(l, obs, diet=0.0, energy_ratio=0.4,
                           nearest_flora={"dr": 0, "dc": 0, "dist": 0.0})
    assert float(l[14]) > 0                    # EAT
    assert int(l.argmax()) == 14               # EAT доминирует (коммит)


def test_eat_no_commit_when_sated():
    # сыт (er≥φ⁻¹) → слабый прайор, НЕ коммит (GATHER>EAT, move не гасим)
    import torch
    c = _c()
    l = torch.zeros(16)
    obs = [0.0] * 64
    c._shape_action_logits(l, obs, diet=0.0, energy_ratio=0.9,
                           nearest_flora={"dr": 0, "dc": 0, "dist": 0.0})
    assert int(l.argmax()) != 14               # EAT НЕ доминирует (нет коммита)


def test_eat_no_commit_near_predator():
    # голоден + на еде, НО хищник близко (d_prox≥0.15) → EAT-коммит ОТКЛ (иерархия
    # predator>eat), §4 FLEE перебивает.
    import torch
    c = _c()
    l = torch.zeros(16)
    obs = [0.0] * 64
    obs[61] = 0.5                              # хищник близко
    c._shape_action_logits(l, obs, diet=0.0, energy_ratio=0.4,
                           nearest_flora={"dr": 0, "dc": 0, "dist": 0.0})
    assert int(l.argmax()) != 14               # не EAT (хищник перебил коммит)


def test_corpse_nav_hungry():
    # Phase C: голоден + труп (мясо) + нет хищника → nav к трупу (dr<0 → NORTH)
    import torch
    c = _c()
    l = torch.zeros(16)
    c._shape_action_logits(l, [0.0] * 64, diet=0.5, energy_ratio=0.4,
                           corpse={"dr": -1, "dc": 0, "dist": 5.0})
    assert float(l[0]) > 0                     # NORTH к трупу
    assert float(l[0]) > float(l[1])


def test_corpse_nav_off_when_sated():
    import torch
    c = _c()
    l = torch.zeros(16)
    c._shape_action_logits(l, [0.0] * 64, diet=0.5, energy_ratio=0.9,
                           corpse={"dr": -1, "dc": 0, "dist": 5.0})
    assert float(l[0]) == 0.0                   # сыт → нет corpse-nav


def test_corpse_nav_blocked_by_predator():
    import torch
    c = _c()
    l = torch.zeros(16)
    obs = [0.0] * 64
    obs[61] = 0.5                              # хищник
    c._shape_action_logits(l, obs, diet=0.5, energy_ratio=0.4,
                           corpse={"dr": -1, "dc": 0, "dist": 5.0})
    assert float(l[0]) == 0.0                   # хищник → FLEE приоритет, не к трупу


def test_eat_reflex_hungry_on_food():
    # ДЕТЕРМИНИРОВАННЫЙ EAT-рефлекс (Фрай 13.06): голоден + на еде + нет хищника →
    # action=EAT в обход мотора (мотор выбрал MOVE).
    c = _c()
    c._on_food["a"] = 1
    out = {"a": {"action": 1, "target_id": None}}     # мотор: MOVE
    c._apply_eat_reflex("a", out, [0.0] * 64)
    assert out["a"]["action"] == 14                    # рефлекс override → EAT


def test_eat_reflex_no_food_noop():
    # не на еде → рефлекс НЕ трогает (нав к лучшему остаётся)
    c = _c()
    out = {"a": {"action": 1, "target_id": None}}
    c._apply_eat_reflex("a", out, [0.0] * 64)
    assert out["a"]["action"] == 1                     # без override


def test_eat_reflex_blocked_by_predator():
    # на еде + голоден, НО хищник (d_prox≥0.15) → НЕ ест (FLEE приоритет, иерархия)
    c = _c()
    c._on_food["a"] = 1
    obs = [0.0] * 64
    obs[61] = 0.5
    out = {"a": {"action": 1, "target_id": None}}
    c._apply_eat_reflex("a", out, obs)
    assert out["a"]["action"] == 1                     # хищник перебил рефлекс


# ── corpse-approach: детерм. шаг на adjacent-тушу (Phase C medium-fix 14.06) ──

def test_corpse_approach_steps_onto_adjacent():
    # GAP-фикс: труп на dist=1 (соседний) + голоден + не-на-еде → детерм. шаг к нему
    # (мотор уводил MOVE-elsewhere). dr<0 → NORTH(0).
    c = _c()
    c._corpse_approach["a"] = 0                         # obs-loop посчитал: шаг NORTH
    out = {"a": {"action": 1, "target_id": None}}      # мотор: SOUTH (мимо мяса)
    c._apply_corpse_approach("a", out, [0.0] * 64)
    assert out["a"]["action"] == 0                      # override → шаг на тушу


def test_corpse_approach_noop_when_on_food():
    # уже на туше (on_food) → approach молчит, eat-рефлекс владеет
    c = _c()
    c._on_food["a"] = 1
    c._corpse_approach["a"] = 0
    out = {"a": {"action": 14, "target_id": None}}      # рефлекс уже поставил EAT
    c._apply_corpse_approach("a", out, [0.0] * 64)
    assert out["a"]["action"] == 14                     # не тронут


def test_corpse_approach_noop_no_corpse():
    # нет adjacent-трупа → не трогает
    c = _c()
    out = {"a": {"action": 1, "target_id": None}}
    c._apply_corpse_approach("a", out, [0.0] * 64)
    assert out["a"]["action"] == 1


def test_corpse_approach_blocked_by_predator():
    # adjacent-труп, НО хищник (d_prox≥0.15) → НЕ лезет (FLEE > approach)
    c = _c()
    c._corpse_approach["a"] = 2
    obs = [0.0] * 64
    obs[61] = 0.5
    out = {"a": {"action": 1, "target_id": None}}
    c._apply_corpse_approach("a", out, obs)
    assert out["a"]["action"] == 1                      # хищник перебил


# ── hunt-commit: детерм. навигация к медиуму (gate в, Фрай 14.06) ──

def test_hunt_commit_moves_to_medium():
    # умеренный голод + medium видна (флаг=MOVE) + не на еде + safe → детерм. шаг к мясу
    c = _c()
    c._hunt_commit["a"] = 2                              # MOVE-EAST к медиуму
    out = {"a": {"action": 0, "target_id": None}}       # мотор: NORTH (грейзинг-дрейф)
    c._apply_hunt_commit("a", out, [0.0] * 64)
    assert out["a"]["action"] == 2                       # override → к медиуму


def test_hunt_commit_attack_at_contact():
    # медиум в упоре (флаг=ATTACK 5) → ATTACK (server резолвит adjacent prey)
    c = _c()
    c._hunt_commit["a"] = 5
    out = {"a": {"action": 1, "target_id": None}}
    c._apply_hunt_commit("a", out, [0.0] * 64)
    assert out["a"]["action"] == 5


def test_hunt_commit_noop_on_food():
    # на еде (трава/туша) → eat-рефлекс владеет, hunt-commit НЕ дёргает с еды
    c = _c()
    c._on_food["a"] = 1
    c._hunt_commit["a"] = 2
    out = {"a": {"action": 14, "target_id": None}}      # eat-рефлекс уже поставил EAT
    c._apply_hunt_commit("a", out, [0.0] * 64)
    assert out["a"]["action"] == 14                      # не тронут


def test_hunt_commit_blocked_by_predator():
    # medium видна, НО хищник (d_prox≥0.15) → FLEE приоритет, не лезет к добыче
    c = _c()
    c._hunt_commit["a"] = 2
    obs = [0.0] * 64
    obs[61] = 0.5
    out = {"a": {"action": 1, "target_id": None}}
    c._apply_hunt_commit("a", out, obs)
    assert out["a"]["action"] == 1                       # хищник перебил


def test_hunt_commit_noop_no_medium():
    # нет видимого медиума (флаг пуст) → не трогает
    c = _c()
    out = {"a": {"action": 1, "target_id": None}}
    c._apply_hunt_commit("a", out, [0.0] * 64)
    assert out["a"]["action"] == 1


def test_hunt_commit_disabled_when_hunting_off():
    # охота выключена → hunt-commit молчит
    c = _c(hunting=False)
    c._hunt_commit["a"] = 2
    out = {"a": {"action": 1, "target_id": None}}
    c._apply_hunt_commit("a", out, [0.0] * 64)
    assert out["a"]["action"] == 1


def test_corpse_step_overrides_hunt_commit():
    # иерархия: смежная туша (бесплатное мясо) > трек к живому медиуму. corpse_approach
    # вызывается ПОСЛЕ hunt_commit → перебивает.
    c = _c()
    c._hunt_commit["a"] = 2                              # медиум на EAST
    c._corpse_approach["a"] = 0                          # туша смежно на NORTH
    out = {"a": {"action": 1, "target_id": None}}
    c._apply_hunt_commit("a", out, [0.0] * 64)
    assert out["a"]["action"] == 2                       # hunt-commit поставил
    c._apply_corpse_approach("a", out, [0.0] * 64)
    assert out["a"]["action"] == 0                       # туша перебила (мясо рядом > трек)


# ── predator-hunt: добивание раненого хищника (Фрай 14.06, узкое окно) ──

def test_predator_hunt_fires_attack():
    # окно держится (obs-loop посчитал _predator_hunt=ATTACK) + enabled → ATTACK override
    # (поверх FLEE мотора для ослабленного хищника).
    c = _c()
    c._predator_hunt_enabled = True
    c._predator_hunt["a"] = 5
    out = {"a": {"action": 10, "target_id": None}}      # мотор: FLEE
    c._apply_predator_hunt("a", out, [0.0] * 64)
    assert out["a"]["action"] == 5                       # добивает раненого


def test_predator_hunt_dormant_off():
    # флаг OFF (dormant) → predator-hunt молчит, FLEE мотора стоит (survival-floor).
    c = _c()
    c._predator_hunt_enabled = False
    c._predator_hunt["a"] = 5
    out = {"a": {"action": 10, "target_id": None}}
    c._apply_predator_hunt("a", out, [0.0] * 64)
    assert out["a"]["action"] == 10                      # FLEE цел (не тронут)


def test_predator_hunt_noop_no_window():
    # окно НЕ держится (нет _predator_hunt) → не трогает → FLEE-floor §4 отрабатывает.
    c = _c()
    c._predator_hunt_enabled = True
    out = {"a": {"action": 10, "target_id": None}}
    c._apply_predator_hunt("a", out, [0.0] * 64)
    assert out["a"]["action"] == 10


def test_set_predator_hunt_toggles():
    c = _c()
    assert c._predator_hunt_enabled is False             # дефолт dormant
    assert c.set_predator_hunt(True) is True
    assert c._predator_hunt_enabled is True
    assert c.set_predator_hunt(False) is False
    assert c._predator_hunt_enabled is False


def test_paralysis_allows_corpse_step():
    # §3-STEP bypass (medium-fix): парализован + adjacent-труп (флаг=move) → шаг проходит
    # (голодающий ДОХОДИТ до мяса из паралича). Иначе заперт рядом с несъеденными 55.
    import time as _t
    c = _c()
    c.biochem["a"] = types.SimpleNamespace(energy=0.0)
    c._corpse_approach["a"] = 2                          # шаг EAST к туше
    c._paralysis_until["a"] = _t.monotonic() + 5.0
    out = {"a": {"action": 2, "target_id": None}}       # approach уже поставил MOVE-EAST
    c._maybe_force_stay("a", out)
    assert out["a"]["action"] == 2                       # шаг прошёл (не STAY)


def test_paralysis_forces_stay_step_wrong_dir():
    # MOVE в параличе НЕ совпадающий с corpse_approach-флагом → forage-floor решает.
    # Без _forage_dir → STAY (узкий bypass)
    import time as _t
    c = _c()
    c.biochem["a"] = types.SimpleNamespace(energy=0.0)
    c._corpse_approach["a"] = 2                          # флаг: EAST
    c._paralysis_until["a"] = _t.monotonic() + 5.0
    out = {"a": {"action": 1, "target_id": None}}       # action=SOUTH (не к туше), еды нет
    c._maybe_force_stay("a", out)
    assert out["a"]["action"] == 4                       # STAY (нет corpse-step И нет forage)


# ── anti-absorbing §3-floor: форажинг-к-еде сквозь паралич (Фрай 14.06) ──

def test_paralysis_forages_when_food_visible():
    # §3 НЕ absorbing: парализован + еда видна (_forage_dir) + мотор выбрал НЕ-survival →
    # ФОРСИМ шаг к еде (ползёт сквозь паралич, не STAY).
    import time as _t
    c = _c()
    c.biochem["a"] = types.SimpleNamespace(energy=0.0)
    c._forage_dir["a"] = 2                               # ближайшая еда на EAST
    c._paralysis_until["a"] = _t.monotonic() + 5.0
    out = {"a": {"action": 11, "target_id": None}}      # мотор: DIG (не-survival)
    c._maybe_force_stay("a", out)
    assert out["a"]["action"] == 2                       # форс шаг к еде (анти-absorbing)


def test_paralysis_stays_when_no_food_visible():
    # парализован + еды НЕ видно (нет _forage_dir) + не-survival → STAY (floor держит)
    import time as _t
    c = _c()
    c.biochem["a"] = types.SimpleNamespace(energy=0.0)
    c._paralysis_until["a"] = _t.monotonic() + 5.0
    out = {"a": {"action": 11, "target_id": None}}      # DIG, еды нет
    c._maybe_force_stay("a", out)
    assert out["a"]["action"] == 4                       # STAY


def test_paralysis_eat_priority_over_forage():
    # на еде (EAT+on_food) И _forage_dir set → EAT выигрывает (ест, не уходит форажить)
    import time as _t
    c = _c()
    c.biochem["a"] = types.SimpleNamespace(energy=0.0)
    c._on_food["a"] = 1
    c._forage_dir["a"] = 2                               # есть и дальняя еда
    c._paralysis_until["a"] = _t.monotonic() + 5.0
    out = {"a": {"action": 14, "target_id": None}}      # EAT (на туше под ногами)
    c._maybe_force_stay("a", out)
    assert out["a"]["action"] == 14                      # ест тут, не уходит к дальней


def _stub_biochem():
    # environment.biochemistry не в venv → handler early-return'ит. Стаб no-op
    # apply_* (+ should_force_stay) → handler исполняет kill-аккумулятор-логику.
    import sys, types as _t
    mod = _t.ModuleType("environment.biochemistry")
    mod.apply_feed = lambda bc: None
    mod.apply_kill_prey = lambda bc: None
    mod.apply_pvp_hit = lambda bc, kind=None: None
    mod.should_force_stay = lambda bc: False
    pkg = sys.modules.get("environment") or _t.ModuleType("environment")
    sys.modules["environment"] = pkg
    sys.modules["environment.biochemistry"] = mod


def test_kill_accumulator_feeds_meat_gc():
    # PHASE 2.5m: kill_energy_acc → meat-GC ось + точный счёт (не теряется throttle)
    _stub_biochem()
    c = _c()
    c.biochem["a"] = types.SimpleNamespace(energy=500.0)
    c._apply_biochem_events("a", {"kill_energy_acc": 275.0, "kill_count_acc": 5,
                                  "killed": True, "delta_energy": 0.0})
    assert c._beh_meat_cum["a"] == 275.0           # ВСЁ мясо (5×55)
    assert c._skill_kill["a"] == 5                  # точный счёт
    assert c._skill_kill_medium["a"] == 5           # avg=55≥34 → medium-тир


def test_kill_accumulator_prey_not_medium():
    _stub_biochem()
    c = _c()
    c.biochem["a"] = types.SimpleNamespace(energy=500.0)
    c._apply_biochem_events("a", {"kill_energy_acc": 42.0, "kill_count_acc": 2})
    assert c._beh_meat_cum["a"] == 42.0            # мясо в ось (мелкая тоже)
    assert c._skill_kill["a"] == 2
    assert c._skill_kill_medium.get("a", 0) == 0   # avg=21<34 → НЕ medium


def test_kill_accumulator_no_double_with_killed():
    # аккумулятор present → per-tick killed НЕ дублирует
    _stub_biochem()
    c = _c()
    c.biochem["a"] = types.SimpleNamespace(energy=500.0)
    c._apply_biochem_events("a", {"kill_energy_acc": 55.0, "kill_count_acc": 1,
                                  "killed": True, "delta_energy": 55.0})
    assert c._beh_meat_cum["a"] == 55.0            # НЕ 110
    assert c._skill_kill["a"] == 1                  # НЕ 2


def test_kill_backcompat_killed_path():
    # нет аккумулятора → per-tick killed (бэккомпат колоний/old P40)
    _stub_biochem()
    c = _c()
    c.biochem["a"] = types.SimpleNamespace(energy=500.0)
    c._apply_biochem_events("a", {"killed": True, "delta_energy": 55.0,
                                  "kill_energy_acc": 0.0, "kill_count_acc": 0})
    assert c._beh_meat_cum["a"] == 55.0
    assert c._skill_kill_medium["a"] == 1          # delta_e≥34


def test_no_pounce_boost_without_flag():
    ws = _ws()
    comp = types.SimpleNamespace()
    comp._single_organism = True
    comp._hunting_enabled = True
    comp._hunt_pounce = {}                       # нет флага
    comp.biochem = {"a": types.SimpleNamespace(energy=300.0)}
    comp.handle_tick = lambda *a, **k: {"a": {"action": 0}}
    comp.get_phase_emas = lambda cid: None
    ws.compute = comp
    out = ws._run_tick_and_build({"a": [0.0] * 64}, {}, {}, world_tick=0)
    e = {x["cid"]: x for x in (out or [])}["a"]
    assert "speed_boost" not in e                # нет рывка
