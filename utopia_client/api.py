"""HTTP-клиент для VPS (divisci.com)."""

from __future__ import annotations

import logging
import requests

logger = logging.getLogger("utopia_client.api")


class UtopiaAPI:
    def __init__(self, server: str, token: str, timeout: float = 15.0) -> None:
        self.server = server.rstrip("/")
        self.token = token
        self.timeout = timeout

    def push_stats(
        self,
        name: str,
        n_alive: int,
        best_fitness: float,
        generation: int,
        world_tick: int,
        extra: dict | None = None,
    ) -> bool:
        url = f"{self.server}/api/colony/stats"
        payload = {
            "name": name,
            "n_alive": int(n_alive),
            "best_fitness": float(best_fitness),
            "generation": int(generation),
            "world_tick": int(world_tick),
            "extra": extra or {},
        }
        try:
            r = requests.post(
                url, json=payload,
                headers={"X-Push-Token": self.token},
                timeout=self.timeout,
            )
            if r.status_code == 200:
                return True
            logger.warning("push_stats HTTP %d: %s", r.status_code, r.text[:200])
            return False
        except Exception as e:
            logger.warning("push_stats error: %s", e)
            return False

    def fetch_command(self) -> dict | None:
        """Опросить желаемое состояние от VPS. None при ошибке."""
        url = f"{self.server}/api/colony/command"
        try:
            r = requests.get(
                url,
                headers={"X-Push-Token": self.token},
                timeout=self.timeout,
            )
            if r.status_code == 200:
                return r.json()
            logger.warning("fetch_command HTTP %d: %s", r.status_code, r.text[:200])
            return None
        except Exception as e:
            logger.warning("fetch_command error: %s", e)
            return None

    def ack_command(self) -> bool:
        """Сбросить oneshot-команду (restart/update_now/benchmark) на VPS в idle.
        Вызывается клиентом ДО выполнения действия, чтобы после restart/execv
        повторный fetch_command не зациклил процесс на той же команде."""
        url = f"{self.server}/api/colony/command/ack"
        try:
            r = requests.post(
                url,
                headers={"X-Push-Token": self.token},
                timeout=self.timeout,
            )
            if r.status_code == 200:
                return True
            logger.warning("ack_command HTTP %d: %s", r.status_code, r.text[:200])
            return False
        except Exception as e:
            logger.warning("ack_command error: %s", e)
            return False

    def push_benchmark(self, result: dict) -> bool:
        url = f"{self.server}/api/colony/benchmark"
        try:
            r = requests.post(
                url, json=result,
                headers={"X-Push-Token": self.token},
                timeout=self.timeout,
            )
            if r.status_code == 200:
                return True
            logger.warning("push_benchmark HTTP %d: %s", r.status_code, r.text[:200])
            return False
        except Exception as e:
            logger.warning("push_benchmark error: %s", e)
            return False

    def push_diagnostics(self, name: str, diag: dict) -> bool:
        """Послать снимок Phase 1/2/6 метрик обучения на VPS.

        Эндпоинт: `POST /api/colony/diagnostics`. Header `X-Push-Token`.
        `diag` — `LocalColonyCompute.diagnostics()` + поле `colony_name`.
        VPS хранит последний снимок per-colony (см. utopia routes_colony.py).
        """
        url = f"{self.server}/api/colony/diagnostics"
        body = dict(diag)
        body["colony_name"] = str(name)
        try:
            r = requests.post(
                url, json=body,
                headers={"X-Push-Token": self.token},
                timeout=self.timeout,
            )
            if r.status_code == 200:
                return True
            logger.warning("push_diagnostics HTTP %d: %s", r.status_code, r.text[:200])
            return False
        except Exception as e:
            logger.warning("push_diagnostics error: %s", e)
            return False

    def push_log_tail(self, lines: list[str]) -> bool:
        """Послать последние строки лога в VPS (admin сможет их прочитать)."""
        if not lines:
            return True
        url = f"{self.server}/api/colony/log_tail"
        try:
            r = requests.post(
                url, json={"lines": lines},
                headers={"X-Push-Token": self.token},
                timeout=self.timeout,
            )
            if r.status_code == 200:
                return True
            logger.warning("push_log_tail HTTP %d: %s", r.status_code, r.text[:200])
            return False
        except Exception as e:
            logger.warning("push_log_tail error: %s", e)
            return False

    def get_client_info(self, channel: str | None = None) -> dict | None:
        """Метаданные актуальной версии клиента: {version, sha256, size, ...}.

        channel (alpha/beta/gamma, схема версий Шефа 17.06): клиент шлёт свой канал
        → сервер отдаёт версию ДЛЯ КАНАЛА (alpha=передовое/main HEAD, beta=
        promoted-эталон, gamma=позже). None → backward-compat (как alpha). Сервер
        игнорит неизвестный param до реализации per-channel (routes_client.py, Хьюберт)."""
        url = f"{self.server}/api/client/info"
        if channel:
            url += f"?channel={channel}"
        try:
            r = requests.get(url, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
            logger.warning("get_client_info HTTP %d", r.status_code)
            return None
        except Exception as e:
            logger.warning("get_client_info error: %s", e)
            return None

    def download_client_zip(self, dest_path: str, channel: str | None = None) -> bool:
        """Скачать актуальный client zip в файл. Для self-update. channel = канал
        версий (alpha/beta/gamma) — zip ДЛЯ КАНАЛА (парный к get_client_info)."""
        url = f"{self.server}/api/client/download"
        if channel:
            url += f"?channel={channel}"
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
            return True
        except Exception as e:
            logger.warning("download_client_zip error: %s", e)
            return False

    def get_neurocore_info(self) -> dict | None:
        """Метаданные актуального neurocore-зеркала: {version, sha256, ...}.

        Используется автоапдейтом зависимости: клиент сравнивает remote sha256
        с записанным локально и при расхождении переустанавливает пакет.
        """
        url = f"{self.server}/api/client/neurocore.json"
        try:
            r = requests.get(url, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
            logger.warning("get_neurocore_info HTTP %d", r.status_code)
            return None
        except Exception as e:
            logger.warning("get_neurocore_info error: %s", e)
            return None

    def get_identity(self) -> dict | None:
        """Имя колонии из кабинета юзера. None при ошибке/невалидном токене.

        VPS-эндпоинт `GET /api/colony/identity` (auth: X-Push-Token) отдаёт
        `{colony_name, email}`. Используется на старте daemon вместо
        локального prompt'а имени в config.json — источник истины переехал
        в `profile.name` на divisci.com (раздел Кабинет → имя).
        """
        url = f"{self.server}/api/colony/identity"
        try:
            r = requests.get(
                url,
                headers={"X-Push-Token": self.token},
                timeout=self.timeout,
            )
            if r.status_code == 200:
                return r.json()
            logger.warning("get_identity HTTP %d: %s", r.status_code, r.text[:200])
            return None
        except Exception as e:
            logger.warning("get_identity error: %s", e)
            return None

    def get_genepool_info(self) -> dict | None:
        """Phase F.7.4: статистика донорского пула Мира (без auth).

        Возвращает {fresh_available, donor_available, donor_pool_size,
        donor_avg_score, world_population, ...} или None при ошибке.
        Используется CLI-диалогом первого подключения для выбора режима
        genesis ('fresh' | 'donor').
        """
        url = f"{self.server}/api/world/genepool/info"
        try:
            r = requests.get(url, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
            logger.warning("get_genepool_info HTTP %d", r.status_code)
            return None
        except Exception as e:
            logger.warning("get_genepool_info error: %s", e)
            return None

    def trigger_p40_lineage_upgrade(
        self, p40_url: str, user_id: str, timeout: float = 5.0,
    ) -> dict | None:
        """Z7.i.b: дёрнуть P40 endpoint Z7.g напрямую по LAN.

        Endpoint: `POST /api/world/upgrade_lineage_to_zodchiy?user_id=<uid>`.
        На стороне P40 (см. neurocore/server/routes_world.py) под `_world_lock`
        выбирается самый энергичный *серверный* Странник владельца и помечается
        `pending_upgrade_to_zodchiy=True` (Z7.f при следующей репродукции
        превратит его потомка в Зодчего).

        У P40 нет VPS-auth, user_id передаётся явным query-параметром
        (резолвится из identity-endpoint VPS, см. main._fetch_colony_name).

        Возвращает JSON ответа (dict) при 2xx, None при ошибке/таймауте/404.
        Не падает: ошибка — soft warning в лог, fallback через client_flags
        polling сохраняется (флаг VPS останется до явного сброса).
        """
        if not user_id:
            logger.warning("trigger_p40_lineage_upgrade: empty user_id")
            return None
        url = f"{p40_url.rstrip('/')}/api/world/upgrade_lineage_to_zodchiy"
        try:
            r = requests.post(
                url, params={"user_id": user_id}, timeout=timeout,
            )
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    return {}
            logger.warning(
                "trigger_p40_lineage_upgrade HTTP %d: %s",
                r.status_code, r.text[:200])
            return None
        except Exception as e:
            logger.warning("trigger_p40_lineage_upgrade error: %s", e)
            return None

    def tag_adam(
        self, p40_url: str, cid: str, timeout: float = 5.0,
    ) -> dict | None:
        """Single-organism pivot (ТЗ e3cc81b, этап 1): пометить cid как Адама.

        Endpoint (Хьюберт, HEAD 33025c6): `POST /api/world/adam/tag`
        body `{cid}` → `{adam_cid, owner_user_id, previous_cid,
        passive_flora_eating}`. Side-effects на P40: is_adam на CreatureState +
        OwnedProjection, глобал _adam_cid, авто passive_flora_eating=True, сброс
        предыдущего tag (один Адам), immortality (7-chokepoint + §3 paralysis)
        с момента tag.

        КРИТИЧНО (порядок, иначе 400/404): cid должен УЖЕ существовать как
        owned-zodchiy проекция в Мире (cid в world + owner != пусто +
        lineage=="zodchiy"). Поэтому сначала спроецировать cid через
        projection_batch (P40 self-heal'ит unknown cid в owned baseline), и
        только ПОТОМ дёргать этот вызов.

        Возвращает JSON при 200, None при ошибке/таймауте. Не падает.
        """
        if not cid:
            logger.warning("tag_adam: empty cid")
            return None
        url = f"{p40_url.rstrip('/')}/api/world/adam/tag"
        try:
            r = requests.post(url, json={"cid": str(cid)}, timeout=timeout)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    return {}
            logger.warning("tag_adam HTTP %d: %s", r.status_code, r.text[:200])
            return None
        except Exception as e:
            logger.warning("tag_adam error: %s", e)
            return None

    def fetch_seed(self, dest_path: str) -> bool:
        return self._fetch_seed_url(f"{self.server}/api/seed", dest_path,
                                    label="seed")

    def fetch_wanderer_seed(self, dest_path: str) -> bool:
        """Скачать wanderer.norg (17 тканей) с VPS."""
        return self._fetch_seed_url(f"{self.server}/api/seed/wanderer",
                                    dest_path, label="wanderer_seed")

    def _fetch_seed_url(self, url: str, dest_path: str, *, label: str) -> bool:
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
            return True
        except Exception as e:
            logger.warning("%s error: %s", label, e)
            return False
