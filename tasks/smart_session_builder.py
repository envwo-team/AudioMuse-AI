"""Smart Listening Sessions request handling and candidate preview helpers."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


SMART_SESSION_MIN_LENGTH = 5
SMART_SESSION_DEFAULT_LENGTH = 25
SMART_SESSION_MAX_LENGTH = 100
SMART_SESSION_DEFAULT_MAX_PER_ARTIST = 2
SMART_SESSION_MAX_ANCHORS = 5
SMART_SESSION_CANDIDATE_POOL_LIMIT = 500

INTENT_WEIGHT = 0.45
ANCHOR_WEIGHT = 0.30
CURVE_WEIGHT = 0.20
DIVERSITY_WEIGHT = 0.05

PROMPT_SOURCE = "prompt"
SEM_GROVE_SOURCE = "sem_grove"
VOYAGER_SOURCE = "voyager"

_CANDIDATE_METADATA_FIELDS = (
    "album_artist",
    "tempo",
    "key",
    "scale",
    "mood_vector",
    "energy",
    "other_features",
    "year",
    "rating",
    "file_path",
)


class SmartSessionValidationError(ValueError):
    """Raised when a Smart Listening Sessions request is invalid."""


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_key(value: Any) -> str:
    return _clean_text(value).casefold()


def _clamp_score(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, score))


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_feature_scores(value: Any) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    if not value:
        return scores
    for pair in str(value).split(","):
        if ":" not in pair:
            continue
        label, raw_score = pair.split(":", 1)
        score = _safe_float(raw_score.strip())
        if score is not None:
            scores[_normalize_key(label)] = _clamp_score(score)
    return scores


def _normalize_tempo(value: Any) -> Optional[float]:
    tempo = _safe_float(value)
    if tempo is None:
        return None
    try:
        from config import TEMPO_MAX_BPM, TEMPO_MIN_BPM
    except Exception:
        TEMPO_MIN_BPM = 40.0
        TEMPO_MAX_BPM = 200.0
    tempo_range = TEMPO_MAX_BPM - TEMPO_MIN_BPM
    if tempo_range <= 0:
        return None
    return _clamp_score((tempo - TEMPO_MIN_BPM) / tempo_range)


def _normalize_energy(value: Any) -> Optional[float]:
    energy = _safe_float(value)
    if energy is None:
        return None
    try:
        from config import ENERGY_MAX, ENERGY_MIN
    except Exception:
        ENERGY_MIN = 0.01
        ENERGY_MAX = 0.15
    if 0.0 <= energy <= 1.0 and energy > ENERGY_MAX:
        return energy
    energy_range = ENERGY_MAX - ENERGY_MIN
    if energy_range <= 0:
        return None
    return _clamp_score((energy - ENERGY_MIN) / energy_range)


def _energy_proxy(candidate: Dict[str, Any]) -> Optional[float]:
    values: List[float] = []
    energy = _normalize_energy(candidate.get("energy"))
    tempo = _normalize_tempo(candidate.get("tempo"))
    if energy is not None:
        values.append(energy)
    if tempo is not None:
        values.append(tempo)

    feature_scores = _parse_feature_scores(candidate.get("other_features"))
    for label in ("aggressive", "party", "danceable", "happy"):
        if label in feature_scores:
            values.append(feature_scores[label])
    for label in ("relaxed", "sad"):
        if label in feature_scores:
            values.append(1.0 - feature_scores[label])

    if not values:
        return None
    return _clamp_score(sum(values) / len(values))


def _score_from_result(row: Dict[str, Any]) -> float:
    if row.get("similarity") is not None:
        return _clamp_score(row.get("similarity"))
    if row.get("score") is not None:
        return _clamp_score(row.get("score"))
    if row.get("distance") is not None:
        try:
            return _clamp_score(1.0 - float(row.get("distance")))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _candidate_pool_size(length: int) -> int:
    return min(SMART_SESSION_CANDIDATE_POOL_LIMIT, max(length * 8, 100))


def _fetch_score_metadata(item_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    item_ids = [item_id for item_id in item_ids if item_id]
    if not item_ids:
        return {}
    try:
        from app_helper import get_score_data_by_ids

        rows = get_score_data_by_ids(item_ids)
        return {row["item_id"]: row for row in rows if row.get("item_id")}
    except Exception:
        logger.exception("Smart Sessions metadata lookup failed")
        return {}


def _candidate_from_result(
    row: Dict[str, Any],
    source: str,
    *,
    anchor: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    item_id = _clean_text(row.get("item_id"))
    if not item_id:
        return None

    metadata = metadata or {}
    title = _clean_text(row.get("title") or metadata.get("title"))
    author = _clean_text(row.get("author") or metadata.get("author"))
    album = _clean_text(row.get("album") or metadata.get("album"))
    score = _score_from_result(row)

    source_scores = {source: score}
    intent_score = score if source == PROMPT_SOURCE else 0.0
    anchor_score = 0.0
    anchor_item_ids: List[str] = []

    if source in {SEM_GROVE_SOURCE, VOYAGER_SOURCE}:
        weight = float(anchor.get("weight", 1.0)) if anchor else 1.0
        anchor_score = _clamp_score(score * weight)
        if anchor and anchor.get("item_id"):
            anchor_item_ids.append(anchor["item_id"])

    return {
        "item_id": item_id,
        "title": title,
        "author": author,
        "album": album,
        "intent_score": intent_score,
        "anchor_score": anchor_score,
        "source_scores": source_scores,
        "sources": [source],
        "anchor_item_ids": anchor_item_ids,
        **{field: row.get(field, metadata.get(field)) for field in _CANDIDATE_METADATA_FIELDS},
    }


def _unique_clean_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        raise SmartSessionValidationError("List value must be an array or string.")

    result = []
    seen = set()
    for value in values:
        cleaned = _clean_text(value)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def clamp_session_length(value: Any) -> int:
    if value in (None, ""):
        return SMART_SESSION_DEFAULT_LENGTH
    try:
        length = int(value)
    except (TypeError, ValueError):
        raise SmartSessionValidationError("Session length must be a number.")
    return max(SMART_SESSION_MIN_LENGTH, min(SMART_SESSION_MAX_LENGTH, length))


def clamp_max_per_artist(value: Any) -> int:
    if value in (None, ""):
        return SMART_SESSION_DEFAULT_MAX_PER_ARTIST
    try:
        max_per_artist = int(value)
    except (TypeError, ValueError):
        raise SmartSessionValidationError("Max per artist must be a number.")
    return max(1, min(10, max_per_artist))


def normalize_anchor(anchor: Any) -> Dict[str, Any]:
    if not isinstance(anchor, dict):
        raise SmartSessionValidationError("Each anchor must be an object.")

    anchor_type = _clean_text(anchor.get("type") or "song").casefold()
    if anchor_type != "song":
        raise SmartSessionValidationError("Only song anchors are supported in the first Smart Sessions version.")

    item_id = _clean_text(anchor.get("item_id"))
    if not item_id:
        raise SmartSessionValidationError("Song anchors require an item_id.")

    try:
        weight = float(anchor.get("weight", 1.0))
    except (TypeError, ValueError):
        raise SmartSessionValidationError("Anchor weight must be a number.")
    weight = max(0.0, min(1.0, weight))

    return {
        "type": "song",
        "item_id": item_id,
        "weight": weight,
    }


def normalize_avoid_rules(data: Any) -> Dict[str, List[str]]:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise SmartSessionValidationError("Avoid rules must be an object.")
    return {
        "artists": _unique_clean_list(data.get("artists")),
        "terms": _unique_clean_list(data.get("terms")),
    }


def validate_preview_request(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise SmartSessionValidationError("Request body must be a JSON object.")

    prompt = _clean_text(data.get("prompt"))
    anchors_raw = data.get("anchors") or []
    if not isinstance(anchors_raw, list):
        raise SmartSessionValidationError("Anchors must be an array.")
    if len(anchors_raw) > SMART_SESSION_MAX_ANCHORS:
        raise SmartSessionValidationError(f"A session can use at most {SMART_SESSION_MAX_ANCHORS} anchors.")

    anchors = [normalize_anchor(anchor) for anchor in anchors_raw]
    if not prompt and not anchors:
        raise SmartSessionValidationError("Provide a prompt or at least one song anchor.")

    curve = _clean_text(data.get("curve") or "steady").casefold()
    valid_curves = {"steady", "calm_to_intense", "intense_to_calm", "near_anchor_then_explore"}
    if curve not in valid_curves:
        raise SmartSessionValidationError("Unsupported session curve.")

    return {
        "prompt": prompt,
        "length": clamp_session_length(data.get("length")),
        "curve": curve,
        "anchors": anchors,
        "avoid": normalize_avoid_rules(data.get("avoid")),
        "max_per_artist": clamp_max_per_artist(data.get("max_per_artist")),
        "include_explanations": bool(data.get("include_explanations", True)),
    }


def validate_export_request(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise SmartSessionValidationError("Request body must be a JSON object.")

    playlist_name = _clean_text(data.get("playlist_name"))
    if not playlist_name:
        raise SmartSessionValidationError("Missing playlist_name.")

    track_ids_raw = data.get("track_ids") or []
    if not isinstance(track_ids_raw, list):
        raise SmartSessionValidationError("track_ids must be an array.")

    track_ids = []
    seen = set()
    for value in track_ids_raw:
        item_id = _clean_text(value)
        if item_id and item_id not in seen:
            seen.add(item_id)
            track_ids.append(item_id)

    if not track_ids:
        raise SmartSessionValidationError("At least one track ID is required.")

    return {
        "playlist_name": playlist_name,
        "track_ids": track_ids,
    }


def get_smart_session_capabilities() -> Dict[str, Any]:
    from config import CLAP_ENABLED, LYRICS_ENABLED

    clap_cache_loaded = False
    clap_song_count = 0
    try:
        from tasks.clap_text_search import get_cache_stats
        clap_stats = get_cache_stats()
        clap_cache_loaded = bool(clap_stats.get("loaded"))
        clap_song_count = int(clap_stats.get("song_count") or 0)
    except Exception:
        clap_cache_loaded = False

    sem_grove_available = False
    sem_grove_song_count = 0
    try:
        from tasks.sem_grove_manager import get_sem_grove_stats
        sem_grove_stats = get_sem_grove_stats()
        sem_grove_available = bool(sem_grove_stats.get("loaded"))
        sem_grove_song_count = int(sem_grove_stats.get("song_count") or 0)
    except Exception:
        sem_grove_available = False

    return {
        "clap_enabled": bool(CLAP_ENABLED),
        "clap_cache_loaded": clap_cache_loaded,
        "clap_song_count": clap_song_count,
        "sem_grove_available": sem_grove_available,
        "sem_grove_song_count": sem_grove_song_count,
        "lyrics_enabled": bool(LYRICS_ENABLED),
        "min_length": SMART_SESSION_MIN_LENGTH,
        "max_length": SMART_SESSION_MAX_LENGTH,
        "default_length": SMART_SESSION_DEFAULT_LENGTH,
        "default_max_per_artist": SMART_SESSION_DEFAULT_MAX_PER_ARTIST,
        "supported_curves": ["steady", "calm_to_intense", "intense_to_calm", "near_anchor_then_explore"],
    }


def build_smart_session_preview(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    request_data = validate_preview_request(data)
    prompt_name = request_data["prompt"][:40].strip() or "Anchored Session"
    playlist_name = f"Smart Session - {prompt_name}"
    prompt_limit = _candidate_pool_size(request_data["length"])
    anchor_limit = min(SMART_SESSION_CANDIDATE_POOL_LIMIT, max(request_data["length"] * 6, 50))

    prompt_candidates, prompt_warnings = get_prompt_candidates(request_data["prompt"], prompt_limit)
    anchor_candidates, anchor_warnings = get_anchor_candidates(request_data["anchors"], anchor_limit)
    merged_candidates = merge_candidate_sources(prompt_candidates, anchor_candidates)
    filtered_candidates, filter_counts = apply_hard_filters(merged_candidates, request_data["avoid"])

    selected_candidates = rank_session_tracks(request_data, filtered_candidates)
    tracks = [
        _format_preview_track(candidate, index + 1, request_data["include_explanations"])
        for index, candidate in enumerate(selected_candidates)
    ]

    warnings = _dedupe_warnings([*prompt_warnings, *anchor_warnings, *_filter_warnings(filter_counts)])
    if not tracks:
        warnings.append("No Smart Session candidates were available from the enabled sources.")
    elif len(tracks) < request_data["length"]:
        warnings.append(f"Only {len(tracks)} tracks were available after source lookup and filters.")

    return {
        "session_id": None,
        "playlist_name": playlist_name,
        "tracks": tracks,
        "warnings": warnings,
        "request": request_data,
    }


def export_smart_session_playlist(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = validate_export_request(data)
    from tasks.voyager_manager import create_playlist_from_ids

    playlist_id = create_playlist_from_ids(payload["playlist_name"], payload["track_ids"])
    return {
        "message": f"Playlist '{payload['playlist_name']}' created successfully!",
        "playlist_id": playlist_id,
    }


def get_prompt_candidates(prompt: str, limit: int) -> tuple[List[Dict[str, Any]], List[str]]:
    prompt = _clean_text(prompt)
    if not prompt:
        return [], []

    warnings: List[str] = []
    try:
        from config import CLAP_ENABLED

        if not CLAP_ENABLED:
            return [], ["CLAP text search is disabled; prompt matching was skipped."]

        from tasks.clap_text_search import get_cache_stats, search_by_text

        cache_stats = get_cache_stats()
        if not cache_stats.get("loaded"):
            return [], ["CLAP cache is not loaded; prompt matching was skipped."]

        results = search_by_text(prompt, limit=limit)
        candidates = []
        for row in results:
            candidate = _candidate_from_result(row, PROMPT_SOURCE)
            if candidate:
                candidates.append(candidate)
        if not candidates:
            warnings.append("CLAP prompt search returned no candidates.")
        return candidates, warnings
    except Exception:
        logger.exception("Smart Sessions prompt candidate search failed")
        return [], ["Prompt matching is temporarily unavailable."]


def _get_sem_grove_anchor_candidates(
    anchor: Dict[str, Any],
    limit: int,
) -> tuple[List[Dict[str, Any]], List[str], bool]:
    try:
        from tasks.sem_grove_manager import get_sem_grove_stats, search_by_song

        stats = get_sem_grove_stats()
        if not stats.get("loaded"):
            return [], ["SemGrove is not loaded; using available anchor fallback sources."], False

        rows = search_by_song(anchor["item_id"], limit=limit)
        candidates = []
        for row in rows:
            if row.get("is_seed"):
                continue
            candidate = _candidate_from_result(row, SEM_GROVE_SOURCE, anchor=anchor)
            if candidate:
                candidates.append(candidate)
        if candidates:
            return candidates, [], True
        warning = f"SemGrove returned no candidates for anchor {anchor['item_id']}."
        return [], [warning], True
    except Exception:
        logger.exception("Smart Sessions SemGrove anchor search failed")
        return [], ["SemGrove anchor matching is temporarily unavailable."], False


def _get_voyager_anchor_candidates(anchor: Dict[str, Any], limit: int) -> tuple[List[Dict[str, Any]], List[str]]:
    try:
        from tasks.voyager_manager import find_nearest_neighbors_by_vector, get_vector_by_id

        seed_vector = get_vector_by_id(anchor["item_id"])
        if seed_vector is None:
            return [], [f"Voyager fallback could not find a vector for anchor {anchor['item_id']}."]

        rows = find_nearest_neighbors_by_vector(seed_vector, n=limit, eliminate_duplicates=True)
        item_ids = [_clean_text(row.get("item_id")) for row in rows if isinstance(row, dict)]
        metadata = _fetch_score_metadata(item_ids)
        candidates = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if _clean_text(row.get("item_id")) == anchor["item_id"]:
                continue
            candidate = _candidate_from_result(
                row,
                VOYAGER_SOURCE,
                anchor=anchor,
                metadata=metadata.get(_clean_text(row.get("item_id")), {}),
            )
            if candidate:
                candidates.append(candidate)
        if not candidates:
            return [], [f"Voyager fallback returned no candidates for anchor {anchor['item_id']}."]
        return candidates, []
    except Exception:
        logger.exception("Smart Sessions Voyager anchor fallback failed")
        return [], ["Voyager anchor fallback is temporarily unavailable."]


def get_anchor_candidates(anchors: List[Dict[str, Any]], limit: int) -> tuple[List[Dict[str, Any]], List[str]]:
    candidates: List[Dict[str, Any]] = []
    warnings: List[str] = []
    if not anchors:
        return candidates, warnings

    per_anchor_limit = max(1, min(limit, SMART_SESSION_CANDIDATE_POOL_LIMIT))
    for anchor in anchors:
        anchor_candidates, anchor_warnings, sem_grove_loaded = _get_sem_grove_anchor_candidates(anchor, per_anchor_limit)
        candidates.extend(anchor_candidates)
        warnings.extend(anchor_warnings)

        if not anchor_candidates or not sem_grove_loaded:
            fallback_candidates, fallback_warnings = _get_voyager_anchor_candidates(anchor, per_anchor_limit)
            candidates.extend(fallback_candidates)
            warnings.extend(fallback_warnings)

    return candidates, _dedupe_warnings(warnings)


def merge_candidate_sources(*candidate_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for group in candidate_groups:
        for candidate in group:
            item_id = candidate.get("item_id")
            if not item_id:
                continue
            existing = merged.get(item_id)
            if not existing:
                merged[item_id] = {
                    **candidate,
                    "source_scores": dict(candidate.get("source_scores", {})),
                    "sources": list(candidate.get("sources", [])),
                    "anchor_item_ids": list(candidate.get("anchor_item_ids", [])),
                }
                continue

            for field in ("title", "author", "album"):
                if not existing.get(field) and candidate.get(field):
                    existing[field] = candidate[field]

            for field in _CANDIDATE_METADATA_FIELDS:
                if existing.get(field) in (None, "") and candidate.get(field) not in (None, ""):
                    existing[field] = candidate[field]

            existing["intent_score"] = max(existing.get("intent_score", 0.0), candidate.get("intent_score", 0.0))
            existing["anchor_score"] = max(existing.get("anchor_score", 0.0), candidate.get("anchor_score", 0.0))

            for source, score in candidate.get("source_scores", {}).items():
                existing["source_scores"][source] = max(existing["source_scores"].get(source, 0.0), score)
            for source in candidate.get("sources", []):
                if source not in existing["sources"]:
                    existing["sources"].append(source)
            for anchor_id in candidate.get("anchor_item_ids", []):
                if anchor_id not in existing["anchor_item_ids"]:
                    existing["anchor_item_ids"].append(anchor_id)

    return list(merged.values())


def apply_hard_filters(
    candidates: List[Dict[str, Any]],
    avoid: Dict[str, List[str]],
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    avoided_artists = {_normalize_key(artist) for artist in avoid.get("artists", [])}
    avoided_terms = [_normalize_key(term) for term in avoid.get("terms", [])]
    filtered: List[Dict[str, Any]] = []
    counts = {
        "missing_metadata": 0,
        "avoided_artist": 0,
        "avoided_term": 0,
    }

    for candidate in candidates:
        item_id = _clean_text(candidate.get("item_id"))
        title = _clean_text(candidate.get("title"))
        author = _clean_text(candidate.get("author"))
        if not item_id or not title or not author:
            counts["missing_metadata"] += 1
            continue

        if _normalize_key(author) in avoided_artists:
            counts["avoided_artist"] += 1
            continue

        title_key = _normalize_key(title)
        if any(term and term in title_key for term in avoided_terms):
            counts["avoided_term"] += 1
            continue

        filtered.append({**candidate, "item_id": item_id, "title": title, "author": author})

    return filtered, counts


def _dedupe_warnings(warnings: List[str]) -> List[str]:
    result = []
    seen = set()
    for warning in warnings:
        key = _normalize_key(warning)
        if warning and key not in seen:
            seen.add(key)
            result.append(warning)
    return result


def _filter_warnings(counts: Dict[str, int]) -> List[str]:
    warnings = []
    if counts.get("avoided_artist"):
        warnings.append(f"Filtered {counts['avoided_artist']} candidate(s) by avoided artist.")
    if counts.get("avoided_term"):
        warnings.append(f"Filtered {counts['avoided_term']} candidate(s) by avoided title term.")
    if counts.get("missing_metadata"):
        warnings.append(f"Filtered {counts['missing_metadata']} candidate(s) missing required track metadata.")
    return warnings


def _candidate_sort_key(candidate: Dict[str, Any]) -> tuple[float, str, str]:
    final_score = _preview_final_score(candidate)
    return (-final_score, _normalize_key(candidate.get("title")), _normalize_key(candidate.get("item_id")))


def _preview_final_score(candidate: Dict[str, Any]) -> float:
    intent_score = _clamp_score(candidate.get("intent_score"))
    anchor_score = _clamp_score(candidate.get("anchor_score"))
    if intent_score and anchor_score:
        return _clamp_score((intent_score * 0.6) + (anchor_score * 0.4))
    return max(intent_score, anchor_score)


def _position_progress(position: int, total: int) -> float:
    if total <= 1:
        return 0.0
    return _clamp_score((position - 1) / (total - 1))


def _curve_score(candidate: Dict[str, Any], position: int, total: int, curve: str) -> float:
    intent_score = _clamp_score(candidate.get("intent_score"))
    anchor_score = _clamp_score(candidate.get("anchor_score"))
    progress = _position_progress(position, total)

    if curve == "steady":
        return 0.5

    if curve == "near_anchor_then_explore":
        exploration_score = max(intent_score, 1.0 - anchor_score if anchor_score else intent_score)
        return _clamp_score((anchor_score * (1.0 - progress)) + (exploration_score * progress))

    energy = _energy_proxy(candidate)
    if energy is None:
        return 0.5

    target = progress if curve == "calm_to_intense" else 1.0 - progress
    return _clamp_score(1.0 - abs(energy - target))


def _selected_title_artist_keys(selected: List[Dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (_normalize_key(candidate.get("title")), _normalize_key(candidate.get("author")))
        for candidate in selected
    }


def _diversity_penalty(
    candidate: Dict[str, Any],
    selected: List[Dict[str, Any]],
    artist_counts: Optional[Dict[str, int]] = None,
    max_per_artist: int = SMART_SESSION_DEFAULT_MAX_PER_ARTIST,
) -> float:
    if not selected:
        return 0.0

    artist_key = _normalize_key(candidate.get("author"))
    album_key = _normalize_key(candidate.get("album"))
    title_key = _normalize_key(candidate.get("title"))
    artist_counts = artist_counts or {}

    penalty = 0.0
    if artist_key:
        penalty += min(1.0, artist_counts.get(artist_key, 0) / max(1, max_per_artist)) * 0.6
    if album_key and any(_normalize_key(track.get("album")) == album_key for track in selected):
        penalty += 0.25
    if title_key and any(_normalize_key(track.get("title")) == title_key for track in selected):
        penalty += 0.5
    return _clamp_score(penalty)


def _is_duplicate_title_artist(candidate: Dict[str, Any], selected: List[Dict[str, Any]]) -> bool:
    key = (_normalize_key(candidate.get("title")), _normalize_key(candidate.get("author")))
    return key in _selected_title_artist_keys(selected)


def score_candidate_for_position(
    candidate: Dict[str, Any],
    position: int,
    total: int,
    curve: str,
    selected: Optional[List[Dict[str, Any]]] = None,
    artist_counts: Optional[Dict[str, int]] = None,
    max_per_artist: int = SMART_SESSION_DEFAULT_MAX_PER_ARTIST,
) -> Dict[str, float]:
    selected = selected or []
    intent_score = _clamp_score(candidate.get("intent_score"))
    anchor_score = _clamp_score(candidate.get("anchor_score"))
    if not intent_score and not anchor_score:
        source_scores = candidate.get("source_scores", {})
        intent_score = max((_clamp_score(score) for score in source_scores.values()), default=0.0)

    curve_score = _curve_score(candidate, position, total, curve)
    diversity_penalty = _diversity_penalty(candidate, selected, artist_counts, max_per_artist)
    final_score = (
        (intent_score * INTENT_WEIGHT)
        + (anchor_score * ANCHOR_WEIGHT)
        + (curve_score * CURVE_WEIGHT)
        - (diversity_penalty * DIVERSITY_WEIGHT)
    )

    return {
        "intent": _clamp_score(intent_score),
        "anchor": _clamp_score(anchor_score),
        "curve": _clamp_score(curve_score),
        "diversity_penalty": _clamp_score(diversity_penalty),
        "final": _clamp_score(final_score),
    }


def _rank_sort_key(candidate: Dict[str, Any]) -> tuple[float, str, str]:
    scores = candidate.get("score_breakdown", {})
    return (-_clamp_score(scores.get("final")), _normalize_key(candidate.get("title")), _normalize_key(candidate.get("item_id")))


def rank_session_tracks(request_data: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    target_length = request_data["length"]
    curve = request_data["curve"]
    max_per_artist = request_data["max_per_artist"]
    selected: List[Dict[str, Any]] = []
    remaining = list(candidates)
    artist_counts: Dict[str, int] = {}

    for position in range(1, target_length + 1):
        scored_candidates: List[Dict[str, Any]] = []
        for candidate in remaining:
            artist_key = _normalize_key(candidate.get("author"))
            if artist_key and artist_counts.get(artist_key, 0) >= max_per_artist:
                continue
            if _is_duplicate_title_artist(candidate, selected):
                continue

            score_breakdown = score_candidate_for_position(
                candidate,
                position,
                target_length,
                curve,
                selected,
                artist_counts,
                max_per_artist,
            )
            scored_candidates.append({**candidate, "score_breakdown": score_breakdown})

        if not scored_candidates:
            break

        chosen = sorted(scored_candidates, key=_rank_sort_key)[0]
        selected.append(chosen)
        remaining = [candidate for candidate in remaining if candidate.get("item_id") != chosen.get("item_id")]
        artist_key = _normalize_key(chosen.get("author"))
        if artist_key:
            artist_counts[artist_key] = artist_counts.get(artist_key, 0) + 1

    return selected


def _build_reason(candidate: Dict[str, Any]) -> str:
    reasons = []
    if candidate.get("intent_score", 0.0) > 0:
        reasons.append("Matches the prompt")
    if candidate.get("anchor_score", 0.0) > 0:
        reasons.append("similar to a selected seed song")
    if len(reasons) < 2 and candidate.get("score_breakdown", {}).get("curve", 0.0) > 0.5 and _energy_proxy(candidate) is not None:
        reasons.append("fits this point in the requested energy curve")
    if not reasons:
        return "Selected from available similarity candidates."
    return "; ".join(reasons[:2]) + "."


def _format_preview_track(candidate: Dict[str, Any], position: int, include_explanations: bool) -> Dict[str, Any]:
    intent_score = _clamp_score(candidate.get("intent_score"))
    anchor_score = _clamp_score(candidate.get("anchor_score"))
    score_breakdown = candidate.get("score_breakdown") or {
        "intent": intent_score,
        "anchor": anchor_score,
        "curve": 0.0,
        "diversity_penalty": 0.0,
        "final": _preview_final_score(candidate),
    }
    track = {
        "item_id": candidate["item_id"],
        "title": candidate.get("title", ""),
        "author": candidate.get("author", ""),
        "album": candidate.get("album", ""),
        "position": position,
        "scores": {
            "intent": round(_clamp_score(score_breakdown.get("intent")), 4),
            "anchor": round(_clamp_score(score_breakdown.get("anchor")), 4),
            "curve": round(_clamp_score(score_breakdown.get("curve")), 4),
            "diversity_penalty": round(_clamp_score(score_breakdown.get("diversity_penalty")), 4),
            "final": round(_clamp_score(score_breakdown.get("final")), 4),
        },
        "sources": candidate.get("sources", []),
        "source_scores": {key: round(_clamp_score(value), 4) for key, value in candidate.get("source_scores", {}).items()},
    }
    if include_explanations:
        track["reason"] = _build_reason(candidate)
    return track