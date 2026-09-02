"""Semantic ranking over the LiteLLM-hosted skill registry, shared by GET /v1/skills?query= and the skill_search MCP tool."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

from openai import OpenAIError
from pydantic import BaseModel, ConfigDict

from litellm.exceptions import BudgetExceededError

if TYPE_CHECKING:
    from litellm.proxy._types import LiteLLM_SkillsTable, UserAPIKeyAuth
    from litellm.router import Router

DEFAULT_SKILL_SEARCH_TOP_K: Final = 5
MAX_SKILL_SEARCH_TOP_K: Final = 100
"""Matches the ``le=100`` bound GET /v1/skills?query= enforces via FastAPI's Query
validation, so the MCP tool can't return a larger payload than the REST endpoint allows."""

Vector: TypeAlias = tuple[float, ...]


class Embedder(Protocol):
    def __call__(self, texts: Sequence[str]) -> Awaitable[Sequence[Vector]]: ...


@dataclass(frozen=True, slots=True)
class SkillSearchHit:
    skill: LiteLLM_SkillsTable
    score: float


@dataclass(frozen=True, slots=True)
class SkillSearchHits:
    hits: tuple[SkillSearchHit, ...]


@dataclass(frozen=True, slots=True)
class SkillSearchNotConfigured:
    reason: str


@dataclass(frozen=True, slots=True)
class SkillSearchEmbeddingFailed:
    reason: str


SkillSearchOutcome: TypeAlias = SkillSearchHits | SkillSearchNotConfigured | SkillSearchEmbeddingFailed


class _EmbeddingItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    embedding: tuple[float, ...]


class _EmbeddingData(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    data: tuple[_EmbeddingItem, ...]


class SkillSearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str
    display_title: str | None
    description: str | None
    score: float


def skill_search_text(skill: LiteLLM_SkillsTable) -> str:
    return "\n".join(part for part in (skill.display_title, skill.description, skill.instructions) if part)


def skill_search_result(hit: SkillSearchHit) -> SkillSearchResult:
    return SkillSearchResult(
        skill_id=hit.skill.skill_id,
        display_title=hit.skill.display_title,
        description=hit.skill.description,
        score=hit.score,
    )


def cosine_similarity(left: Vector, right: Vector) -> float:
    dot: Final = sum(a * b for a, b in zip(left, right, strict=True))
    norms: Final = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norms if norms else 0.0


def embedding_spend_metadata(user_api_key_dict: UserAPIKeyAuth) -> Mapping[str, object]:
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

    return {  # mutable-ok: the router mutates the metadata dict it is handed
        **LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(user_api_key_dict),
        "user_api_key": user_api_key_dict.api_key,
    }


def router_embedder(router: Router, embedding_model: str, user_api_key_dict: UserAPIKeyAuth) -> Embedder:
    async def embed(texts: Sequence[str]) -> Sequence[Vector]:
        batch: Final = list(texts)  # mutable-ok: Router.aembedding accepts only str | list input
        response: Final = await router.aembedding(
            model=embedding_model, input=batch, metadata=embedding_spend_metadata(user_api_key_dict)
        )
        return tuple(item.embedding for item in _EmbeddingData.model_validate(response.model_dump()).data)

    return embed


_NO_VECTORS: Final[Mapping[str, Vector]] = MappingProxyType({})


async def _embed_all(embed: Embedder, texts: Sequence[str]) -> tuple[Vector, ...] | SkillSearchEmbeddingFailed:
    try:
        vectors: Final = tuple(await embed(texts))
    except (OpenAIError, ValueError, BudgetExceededError) as exc:
        return SkillSearchEmbeddingFailed(reason=f"embedding the search query failed: {exc}")
    if len(vectors) != len(texts):
        return SkillSearchEmbeddingFailed(
            reason=f"embedding model returned {len(vectors)} vectors for {len(texts)} inputs"
        )
    return vectors


@dataclass(frozen=True, slots=True)
class _Embedded:
    query_vector: Vector
    vectors: Mapping[str, Vector]


def _same_dimension(query_vector: Vector, vectors: Mapping[str, Vector], texts: Sequence[str]) -> bool:
    return all(len(vectors[text]) == len(query_vector) for text in texts)


async def _embed_query_and_skills(
    embed: Embedder, query: str, texts: Sequence[str], cached: Mapping[str, Vector]
) -> _Embedded | SkillSearchEmbeddingFailed:
    missing: Final = tuple(dict.fromkeys(text for text in texts if text not in cached))
    embedded: Final = await _embed_all(embed, (query, *missing))
    if isinstance(embedded, SkillSearchEmbeddingFailed):
        return embedded
    vectors: Final = MappingProxyType(dict(chain(cached.items(), zip(missing, embedded[1:], strict=True))))
    if _same_dimension(embedded[0], vectors, texts):
        return _Embedded(query_vector=embedded[0], vectors=vectors)
    unique: Final = tuple(dict.fromkeys(texts))
    reembedded: Final = await _embed_all(embed, (query, *unique))
    if isinstance(reembedded, SkillSearchEmbeddingFailed):
        return reembedded
    return _Embedded(
        query_vector=reembedded[0], vectors=MappingProxyType(dict(zip(unique, reembedded[1:], strict=True)))
    )


class SkillSearchIndex:
    """Caches one vector per distinct skill text per embedding model, so repeat searches only embed the query."""

    def __init__(self) -> None:
        self._vectors: Mapping[str, Mapping[str, Vector]] = MappingProxyType({})

    def _merged(self, embedding_model: str, embedded: _Embedded) -> Mapping[str, Vector]:
        kept: Final = {
            text: vector
            for text, vector in self._vectors.get(embedding_model, _NO_VECTORS).items()
            if len(vector) == len(embedded.query_vector)
        }
        return MappingProxyType({**kept, **embedded.vectors})

    async def search(
        self,
        query: str,
        skills: Sequence[LiteLLM_SkillsTable],
        top_k: int,
        embed: Embedder,
        embedding_model: str,
    ) -> SkillSearchHits | SkillSearchEmbeddingFailed:
        if not skills:
            return SkillSearchHits(hits=())
        texts: Final = tuple(skill_search_text(skill) for skill in skills)
        cached: Final = self._vectors.get(embedding_model, _NO_VECTORS)
        embedded: Final = await _embed_query_and_skills(embed, query, texts, cached)
        if isinstance(embedded, SkillSearchEmbeddingFailed):
            return embedded
        if not _same_dimension(embedded.query_vector, embedded.vectors, texts):
            return SkillSearchEmbeddingFailed(
                reason=f"embedding model {embedding_model} returned vectors of mixed dimensions"
            )
        self._vectors = MappingProxyType({**self._vectors, embedding_model: self._merged(embedding_model, embedded)})
        ranked: Final = sorted(
            (
                SkillSearchHit(skill=skill, score=cosine_similarity(embedded.query_vector, embedded.vectors[text]))
                for skill, text in zip(skills, texts, strict=True)
            ),
            key=lambda hit: hit.score,
            reverse=True,
        )
        return SkillSearchHits(hits=tuple(ranked[:top_k]))


global_skill_search_index: Final = SkillSearchIndex()


async def search_skills(
    query: str,
    skills: Sequence[LiteLLM_SkillsTable],
    top_k: int,
    router: Router | None,
    embedding_model: str | None,
    index: SkillSearchIndex,
    user_api_key_dict: UserAPIKeyAuth,
) -> SkillSearchOutcome:
    if embedding_model is None:
        return SkillSearchNotConfigured(
            reason="skill search needs litellm_settings.skill_search_embedding_model set to an embedding model from model_list"
        )
    if router is None:
        return SkillSearchNotConfigured(reason="skill search needs a model_list so the embedding model can be called")
    return await index.search(
        query, skills, top_k, router_embedder(router, embedding_model, user_api_key_dict), embedding_model
    )
