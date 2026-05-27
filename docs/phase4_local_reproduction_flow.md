# Phase 4 — Local-only reproduction flow (draft для Хьюберта)

**Автор:** Бендер (utopia-client)
**Дата:** 27.05.2026
**Статус:** Draft, нужно ревью + ответы на open questions
**Связь:** [tz_body_migration.md §4 Phase 4 acceptance #3-4, §8 Q3]

## TL;DR

Phase 4 Q3 DECIDED (Шеф 26.05): «Размножение только внутри одной
клиент-популяции. Никакого cross-population matching через брокер».

Текущая реализация **частично client-side** (crossover в
`utopia_client/crossover.py`, asexual mutation в `reproduce.py`), но
**контроль flow остаётся на P40**: P40 инициирует `mate_request`,
client отвечает, P40 регистрирует newborn.

Этот документ предлагает **полный flow shift на client** с минимальными
изменениями на P40 (P40 = пассивный регистратор).

---

## Текущий flow (pre-Phase 4)

### Sexual (mate_pair)

```
P40 tick loop:
  _maybe_mate_pair() → выбирает (mother, father) по valence/proximity
  → if both owned by same client → P40 send mate_request to client
  → client ws_client._handle_mate_request:
       1. load father weights из msg
       2. apply_crossover_inheritance (mother_org + father_org)
       3. serialize_organism_blob(child)
       4. reply child_blob
  → P40 register_newborn(child_blob)
  → newborn_ack to client
  → client add_creature(child_cid, child_org)
```

### Asexual (REPRODUCE-action)

```
LocalColonyCompute.handle_tick:
  motor выбирает REPRODUCE action
  → build_reproduce_envelope(parent_cid, organism)
     - mutate_state_dict (Y50 на weights)
     - [Phase 4 этап C] mutate_topology_genes (opt-in)
     - pack_zstd_b64
  → send "reproduce" envelope to P40 via WS
  → P40 validate, _inherit_traits, register_newborn(child_blob)
  → newborn_ack to client
  → client add_creature
```

**Общее:** в обоих случаях **P40 = central authority** для newborn
registration и cid allocation.

---

## Target flow (Phase 4 Q3 local-only)

### Sexual

```
LocalColonyCompute.handle_tick:
  _detect_mate_pairs(): scan owned organisms для (sex_drive_a >
    threshold AND sex_drive_b > threshold AND distance < radius)
  → on detection:
     1. mother, father = selected pair
     2. child_cid = local_alloc_cid()  # see Q1
     3. clone topology + crossover_topology_genes(mother, father)
     4. apply_crossover_inheritance(child_org, mother, father)
     5. assign_species(registry, child_topology, ...)
     6. add_creature(child_cid, child_org, lineage="zodchiy")
     7. emit embodied/events: newborn_announce {cid: child_cid,
        parent_cids: [mother, father], species_id: ...}
  → P40 receives newborn_announce
  → P40 register_newborn_passive(cid, parent_cids, species_id)
     - inherit physical traits (vision/attack radius) from parents
     - place в world at parent's position
     - не создаёт ColonyMember (тело уже на client)
```

### Asexual

```
LocalColonyCompute.handle_tick:
  motor выбирает REPRODUCE action
  → 1. child_cid = local_alloc_cid()
     2. parent_topology genes → mutate_topology_genes
     3. assign_species(registry, child_topology, ...)
     4. clone tissues + mutate_state_dict (Y50)
     5. add_creature(child_cid, child_org, lineage="zodchiy")
     6. emit newborn_announce {cid, parent_cids: [parent], species_id}
  → P40 register_newborn_passive (тот же handler что sexual)
```

---

## Что нужно от P40 (Хьюбертова работа)

### 1. Новый WS message: `newborn_announce`

**Direction:** client → P40 (через VPS broker).

**Payload:**
```python
{
    "type": "newborn_announce",
    "cid": str,                    # client-allocated
    "parent_cids": [str, ...],     # 1 (asexual) или 2 (sexual)
    "species_id": int,             # client-local species
    "topology_summary": {          # optional, для debug
        "n_genes": int,
        "innovations": [int, ...],
    },
}
```

**P40 handler (`routes_world.py:_handle_newborn_announce`):**
1. Validate parent_cids exist in world (alive).
2. Inherit physical traits через `_inherit_traits` (vision_radius,
   attack_radius, speed_factor — base traits, не tissue weights).
3. Place в world at parent's position.
4. **НЕ** создавать `ColonyMember` — тело только у client.
5. Send ack: `newborn_announce_ack {cid, accepted: bool, reason?}`.

### 2. Stop initiating `mate_request`

P40 `_maybe_mate_pair()` → исключить owned Zodchiy из mate detection.
Это **очищающий cleanup** в Phase 6 (legacy cleanup), но для Phase 4
можно сделать gate'нутый флаг `client_owns_reproduction` (default
False, активируется когда client готов).

### 3. Phase out `register_newborn` для Zodchiy

`register_newborn` остаётся для wanderers/elders, но для Zodchiy —
переключение на `_handle_newborn_announce`. Legacy `mate_request`
flow остаётся как fallback (deprecated).

---

## Open questions (для ответа Шеф/Хьюберт)

### Q1: cid allocation — client или P40?

**Вариант A (Бендер default):** client allocates `cid = uuid.uuid4()`.
- ✅ Простота: не нужен round-trip для newborn
- ⚠️ Collision risk: < 2^-64, но non-zero across clients
- ⚠️ P40 может отказать если cid уже занят

**Вариант B:** P40 allocates cid через `cid_reserve` request.
- Round-trip overhead перед reproduction
- Guaranteed uniqueness
- Лишний WS message

**Рекомендация Бендера:** Вариант A. Collision risk acceptable,
P40 в ack может сказать "rename" если коллизия.

### Q2: race conditions при concurrent newborn

Если 2 client'а в один tick рожают organisms с cid'ами что
коллизуют (Q1.A), P40 ack должен один из них reject. Client
получает `newborn_announce_ack {accepted: false, reason: cid_taken}`
и выполняет cleanup (remove_creature).

**Acceptable** для Phase 4 — collision вероятность мала.

### Q3: backward compat для legacy Zodchiy на P40

Legacy mature Zodchiy созданные до Phase 4 — у них тело на P40.
Они могут участвовать в reproduction по старому mate_request flow
(пока не мигрированы в Phase 6).

**Решение:** mate_request flow **остаётся** для legacy. Новые
Zodchiy используют newborn_announce flow.

Detection: P40 проверяет в `_maybe_mate_pair` где организм живёт
(legacy ColonyMember vs client-owned). Если owned → skip mate_request,
ждать client newborn_announce.

### Q4: physical traits inheritance

Сейчас P40 делает `_inherit_traits(parent_a, parent_b, child)` —
вычисляет vision/attack/speed/size с σ-шумом.

**Вопрос:** делаем это на client (нужны parent traits как часть
public state) или на P40 (через newborn_announce_ack)?

**Рекомендация Бендера:** на P40. Physical traits = публичная граница
(см. tz_body_migration.md §2.2), client их не контролирует.
Client отправляет newborn_announce с parent_cids, P40 вычисляет
traits и в ack возвращает; client обновляет `traits` field на
newborn organism.

### Q5: speciation cross-validation

Client-local species — каждый client держит свой `SpeciesRegistry`.
Если две client-популяции независимо эволюционируют одинаковые
species (convergent), они будут разными species_id.

Это **acceptable** по Q3 vision §3.7 — никакой кросс-клиентной
дедупликации. Lab snapshot на VPS может показать «species_id=5
на client A» и «species_id=5 на client B» как разные строки.

---

## Готовое client-side (что уже работает)

- `utopia_client/reproduce.py`: build_reproduce_envelope (asexual)
  + Phase 4 этап C mutate_topology opt-in
- `utopia_client/crossover.py`: apply_crossover_inheritance (sexual)
- `utopia_client/speciation.py`: assign_species + persistence (Phase 4 этап D)
- `utopia_client/memory_store.py`: episodic persistence (Phase 4 этап B)
- `utopia_client/embodied_ws.py`: WS канал для embodied events (готов
  принимать новые message types)

## Что нужно сделать в utopia-client (после P40-side ready)

1. `LocalColonyCompute._detect_mate_pairs()` — scan owned cid'ов
2. `LocalColonyCompute._handle_reproduction(parent_cids)` — full flow
3. Wire `assign_species` + `species_id` tracking (task #40)
4. New message handler `newborn_announce_ack` в ws_client
5. Stop using mate_request handler как primary (legacy fallback)
6. Pytest end-to-end: detect → crossover → announce → ack → registered

## ETA

- P40 side (Хьюберт): ~3-5 дней (новый handler + traits + ack flow)
- Client side (Бендер): ~3 дня (после P40 ready)
- Pytest + production verify: 1-2 дня

Total Phase 4 closure: **~1-1.5 недели после старта Хьюберта**.
