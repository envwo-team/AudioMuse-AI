import sys
import types

import pytest

from tasks import smart_session_builder as builder
from tasks.smart_session_builder import (
    SMART_SESSION_DEFAULT_LENGTH,
    SMART_SESSION_MAX_LENGTH,
    SMART_SESSION_MIN_LENGTH,
    SmartSessionValidationError,
    apply_hard_filters,
    build_smart_session_preview,
    get_anchor_candidates,
    get_prompt_candidates,
    merge_candidate_sources,
    normalize_avoid_rules,
    normalize_anchor,
    rank_session_tracks,
    score_candidate_for_position,
    validate_export_request,
    validate_preview_request,
)


def _install_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _candidate(item_id, title, author, intent=0.5, anchor=0.0, **metadata):
    return {
        'item_id': item_id,
        'title': title,
        'author': author,
        'album': metadata.pop('album', ''),
        'intent_score': intent,
        'anchor_score': anchor,
        'source_scores': {'prompt': intent} if intent else {},
        'sources': ['prompt'] if intent else [],
        'anchor_item_ids': [],
        **metadata,
    }


def _request(length=5, curve='steady', max_per_artist=2):
    return {
        'prompt': 'test',
        'length': length,
        'curve': curve,
        'anchors': [],
        'avoid': {'artists': [], 'terms': []},
        'max_per_artist': max_per_artist,
        'include_explanations': True,
    }


def test_preview_requires_prompt_or_anchor():
    with pytest.raises(SmartSessionValidationError, match='prompt'):
        validate_preview_request({'prompt': '   ', 'anchors': []})


def test_preview_clamps_length_and_max_per_artist():
    data = validate_preview_request({
        'prompt': 'quiet synths',
        'length': 999,
        'max_per_artist': 0,
    })

    assert data['length'] == SMART_SESSION_MAX_LENGTH
    assert data['max_per_artist'] == 1


def test_preview_uses_default_length():
    data = validate_preview_request({'prompt': 'morning acoustic'})

    assert data['length'] == SMART_SESSION_DEFAULT_LENGTH


def test_preview_clamps_short_length():
    data = validate_preview_request({'prompt': 'short set', 'length': 1})

    assert data['length'] == SMART_SESSION_MIN_LENGTH


def test_normalize_anchor_accepts_song_anchor():
    anchor = normalize_anchor({'type': 'song', 'item_id': ' track-1 ', 'weight': 1.5})

    assert anchor == {'type': 'song', 'item_id': 'track-1', 'weight': 1.0}


def test_normalize_anchor_rejects_unknown_type():
    with pytest.raises(SmartSessionValidationError, match='Only song anchors'):
        normalize_anchor({'type': 'artist', 'item_id': 'artist-1'})


def test_normalize_avoid_rules_deduplicates_values():
    rules = normalize_avoid_rules({
        'artists': [' Alice ', 'alice', 'Bob'],
        'terms': 'live',
    })

    assert rules == {'artists': ['Alice', 'Bob'], 'terms': ['live']}


def test_get_prompt_candidates_uses_clap_search(monkeypatch):
    _install_module(monkeypatch, 'config', CLAP_ENABLED=True)
    search_calls = []

    def fake_search(prompt, limit):
        search_calls.append((prompt, limit))
        return [{
            'item_id': 'song-1',
            'title': 'Warm Dusk',
            'author': 'Alice',
            'album': 'Evening',
            'similarity': 0.82,
        }]

    _install_module(
        monkeypatch,
        'tasks.clap_text_search',
        get_cache_stats=lambda: {'loaded': True, 'song_count': 1},
        search_by_text=fake_search,
    )

    candidates, warnings = get_prompt_candidates(' warm dusk ', 25)

    assert warnings == []
    assert search_calls == [('warm dusk', 25)]
    assert candidates[0]['item_id'] == 'song-1'
    assert candidates[0]['intent_score'] == pytest.approx(0.82)
    assert candidates[0]['sources'] == ['prompt']


def test_get_prompt_candidates_warns_when_clap_cache_unloaded(monkeypatch):
    _install_module(monkeypatch, 'config', CLAP_ENABLED=True)
    _install_module(
        monkeypatch,
        'tasks.clap_text_search',
        get_cache_stats=lambda: {'loaded': False},
        search_by_text=lambda prompt, limit: pytest.fail('search should not run'),
    )

    candidates, warnings = get_prompt_candidates('ambient focus', 10)

    assert candidates == []
    assert warnings == ['CLAP cache is not loaded; prompt matching was skipped.']


def test_get_anchor_candidates_uses_sem_grove(monkeypatch):
    _install_module(
        monkeypatch,
        'tasks.sem_grove_manager',
        get_sem_grove_stats=lambda: {'loaded': True},
        search_by_song=lambda item_id, limit: [
            {'item_id': item_id, 'title': 'Seed', 'author': 'Seed Artist', 'similarity': 1.0, 'is_seed': True},
            {'item_id': 'near-1', 'title': 'Nearby', 'author': 'Bob', 'album': 'Close', 'similarity': 0.75},
        ],
    )

    candidates, warnings = get_anchor_candidates([{'type': 'song', 'item_id': 'seed-1', 'weight': 0.8}], 10)

    assert warnings == []
    assert [candidate['item_id'] for candidate in candidates] == ['near-1']
    assert candidates[0]['anchor_score'] == pytest.approx(0.6)
    assert candidates[0]['sources'] == ['sem_grove']


def test_get_anchor_candidates_falls_back_to_voyager(monkeypatch):
    _install_module(
        monkeypatch,
        'tasks.sem_grove_manager',
        get_sem_grove_stats=lambda: {'loaded': False},
        search_by_song=lambda item_id, limit: [],
    )
    _install_module(
        monkeypatch,
        'tasks.voyager_manager',
        get_vector_by_id=lambda item_id: [0.1, 0.2],
        find_nearest_neighbors_by_vector=lambda vector, n, eliminate_duplicates: [
            {'item_id': 'seed-1', 'distance': 0.0},
            {'item_id': 'fallback-1', 'distance': 0.2},
        ],
    )
    monkeypatch.setattr(builder, '_fetch_score_metadata', lambda item_ids: {
        'fallback-1': {'title': 'Fallback Song', 'author': 'Cara', 'album': 'Fallback Album'},
    })

    candidates, warnings = get_anchor_candidates([{'type': 'song', 'item_id': 'seed-1', 'weight': 1.0}], 10)

    assert warnings == ['SemGrove is not loaded; using available anchor fallback sources.']
    assert [candidate['item_id'] for candidate in candidates] == ['fallback-1']
    assert candidates[0]['title'] == 'Fallback Song'
    assert candidates[0]['anchor_score'] == pytest.approx(0.8)
    assert candidates[0]['sources'] == ['voyager']


def test_merge_candidate_sources_deduplicates_and_preserves_best_scores():
    prompt_candidates = [{
        'item_id': 'song-1',
        'title': 'Same Song',
        'author': 'Alice',
        'album': '',
        'intent_score': 0.7,
        'anchor_score': 0.0,
        'source_scores': {'prompt': 0.7},
        'sources': ['prompt'],
        'anchor_item_ids': [],
    }]
    anchor_candidates = [{
        'item_id': 'song-1',
        'title': '',
        'author': '',
        'album': 'Merged Album',
        'intent_score': 0.0,
        'anchor_score': 0.9,
        'source_scores': {'sem_grove': 0.9},
        'sources': ['sem_grove'],
        'anchor_item_ids': ['seed-1'],
    }]

    merged = merge_candidate_sources(prompt_candidates, anchor_candidates)

    assert len(merged) == 1
    assert merged[0]['intent_score'] == pytest.approx(0.7)
    assert merged[0]['anchor_score'] == pytest.approx(0.9)
    assert merged[0]['album'] == 'Merged Album'
    assert merged[0]['sources'] == ['prompt', 'sem_grove']


def test_apply_hard_filters_removes_avoids_and_missing_metadata():
    candidates = [
        {'item_id': 'keep', 'title': 'Studio Track', 'author': 'Alice'},
        {'item_id': 'artist', 'title': 'Another Track', 'author': 'Blocked Artist'},
        {'item_id': 'term', 'title': 'Big Remix', 'author': 'Bob'},
        {'item_id': 'missing', 'title': '', 'author': 'Cara'},
    ]

    filtered, counts = apply_hard_filters(candidates, {
        'artists': ['blocked artist'],
        'terms': ['remix'],
    })

    assert [candidate['item_id'] for candidate in filtered] == ['keep']
    assert counts == {'missing_metadata': 1, 'avoided_artist': 1, 'avoided_term': 1}


def test_build_preview_returns_day_two_candidate_tracks(monkeypatch):
    monkeypatch.setattr(builder, 'get_prompt_candidates', lambda prompt, limit: ([{
        'item_id': 'song-1',
        'title': 'Warm Dusk',
        'author': 'Alice',
        'album': 'Evening',
        'intent_score': 0.84,
        'anchor_score': 0.0,
        'source_scores': {'prompt': 0.84},
        'sources': ['prompt'],
        'anchor_item_ids': [],
    }], []))
    monkeypatch.setattr(builder, 'get_anchor_candidates', lambda anchors, limit: ([], []))

    preview = build_smart_session_preview({'prompt': 'warm dusk songs', 'length': 5})

    assert preview['session_id'] is None
    assert preview['playlist_name'] == 'Smart Session - warm dusk songs'
    assert preview['tracks'][0]['item_id'] == 'song-1'
    assert preview['tracks'][0]['scores']['intent'] == pytest.approx(0.84)
    assert preview['tracks'][0]['reason'] == 'Matches the prompt.'
    assert preview['warnings'] == ['Only 1 tracks were available after source lookup and filters.']


def test_score_candidate_for_position_uses_real_energy_for_curves():
    low_energy = _candidate('low', 'Low', 'Alice', energy=0.01)
    high_energy = _candidate('high', 'High', 'Bob', energy=0.15)

    low_start = score_candidate_for_position(low_energy, 1, 3, 'calm_to_intense')
    high_start = score_candidate_for_position(high_energy, 1, 3, 'calm_to_intense')
    low_end = score_candidate_for_position(low_energy, 3, 3, 'calm_to_intense')
    high_end = score_candidate_for_position(high_energy, 3, 3, 'calm_to_intense')

    assert low_start['curve'] > high_start['curve']
    assert high_end['curve'] > low_end['curve']


def test_rank_calm_to_intense_orders_equal_matches_by_energy():
    candidates = [
        _candidate('high', 'High', 'Alice', energy=0.15),
        _candidate('low', 'Low', 'Bob', energy=0.01),
        _candidate('mid', 'Mid', 'Cara', energy=0.08),
    ]

    ranked = rank_session_tracks(_request(length=3, curve='calm_to_intense'), candidates)

    assert [track['item_id'] for track in ranked] == ['low', 'mid', 'high']
    assert ranked[0]['score_breakdown']['curve'] > ranked[-1]['score_breakdown']['curve']


def test_rank_intense_to_calm_reverses_energy_curve():
    candidates = [
        _candidate('low', 'Low', 'Alice', energy=0.01),
        _candidate('high', 'High', 'Bob', energy=0.15),
        _candidate('mid', 'Mid', 'Cara', energy=0.08),
    ]

    ranked = rank_session_tracks(_request(length=3, curve='intense_to_calm'), candidates)

    assert [track['item_id'] for track in ranked] == ['high', 'mid', 'low']


def test_rank_near_anchor_then_explore_starts_with_anchor_match():
    candidates = [
        _candidate('prompt', 'Prompt Fit', 'Alice', intent=0.9, anchor=0.1),
        _candidate('anchor', 'Anchor Fit', 'Bob', intent=0.1, anchor=0.9),
    ]

    ranked = rank_session_tracks(_request(length=2, curve='near_anchor_then_explore'), candidates)

    assert [track['item_id'] for track in ranked] == ['anchor', 'prompt']


def test_rank_respects_max_per_artist():
    candidates = [
        _candidate('a1', 'Artist One', 'Alice', intent=0.9),
        _candidate('a2', 'Artist Two', 'Alice', intent=0.85),
        _candidate('b1', 'Other One', 'Bob', intent=0.6),
        _candidate('c1', 'Other Two', 'Cara', intent=0.55),
    ]

    ranked = rank_session_tracks(_request(length=3, max_per_artist=1), candidates)

    assert [track['item_id'] for track in ranked] == ['a1', 'b1', 'c1']
    assert sum(1 for track in ranked if track['author'] == 'Alice') == 1


def test_rank_tie_breaks_by_title_then_item_id():
    candidates = [
        _candidate('b-id', 'Beta', 'Alice', intent=0.7),
        _candidate('z-id', 'Alpha', 'Bob', intent=0.7),
        _candidate('a-id', 'Alpha', 'Cara', intent=0.7),
    ]

    ranked = rank_session_tracks(_request(length=3), candidates)

    assert [track['item_id'] for track in ranked] == ['a-id', 'z-id', 'b-id']


def test_rank_skips_same_title_same_artist_duplicates():
    candidates = [
        _candidate('one', 'Same Title', 'Alice', intent=0.9),
        _candidate('two', 'Same Title', 'Alice', intent=0.8),
        _candidate('three', 'Different Title', 'Alice', intent=0.7),
    ]

    ranked = rank_session_tracks(_request(length=3, max_per_artist=3), candidates)

    assert [track['item_id'] for track in ranked] == ['one', 'three']


def test_anchor_only_explanation_does_not_claim_prompt_match(monkeypatch):
    monkeypatch.setattr(builder, 'get_prompt_candidates', lambda prompt, limit: ([], []))
    monkeypatch.setattr(builder, 'get_anchor_candidates', lambda anchors, limit: ([{
        'item_id': 'song-1',
        'title': 'Nearby',
        'author': 'Alice',
        'album': 'Evening',
        'intent_score': 0.0,
        'anchor_score': 0.84,
        'source_scores': {'sem_grove': 0.84},
        'sources': ['sem_grove'],
        'anchor_item_ids': ['seed-1'],
    }], []))

    preview = build_smart_session_preview({
        'anchors': [{'type': 'song', 'item_id': 'seed-1'}],
        'length': 5,
        'curve': 'steady',
    })

    assert preview['tracks'][0]['reason'] == 'similar to a selected seed song.'
    assert 'prompt' not in preview['tracks'][0]['reason']


def test_export_validation_deduplicates_track_ids():
    payload = validate_export_request({
        'playlist_name': 'My Session',
        'track_ids': ['a', 'a', ' b ', ''],
    })

    assert payload == {'playlist_name': 'My Session', 'track_ids': ['a', 'b']}