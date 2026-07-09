"""Tests for StrategyRouter."""
from unittest.mock import MagicMock

import pytest

from bookmind_tutor.agents.strategies.router import StrategyRouter


def _make_router():
    react = MagicMock()
    react.name = "react"
    react.chat.return_value = "react answer"
    react.last_trace = None

    plan = MagicMock()
    plan.name = "plan_execute"
    plan.chat.return_value = "plan answer"
    plan.last_trace = None

    reflexion = MagicMock()
    reflexion.name = "reflexion"
    reflexion.chat.return_value = "reflexion answer"
    reflexion.last_trace = None

    router = StrategyRouter(react=react, plan_execute=plan, reflexion=reflexion)
    return router, react, plan, reflexion


@pytest.mark.parametrize("question", [
    "Compare leader-based vs leaderless replication",
    "What is the difference between replication and partitioning?",
    "Contrast eventual vs strong consistency",
    "How do DDIA and Clean Architecture differ in their approach?",
    "Explain both replication and partitioning",
])
def test_plan_keywords_route_to_plan_execute(question):
    router, _, plan, _ = _make_router()
    router.chat(question)
    plan.chat.assert_called_once()


@pytest.mark.parametrize("question", [
    "Explain in depth how consensus algorithms work",
    "Give a detailed explanation of write-ahead logs",
    "Thoroughly explain the CAP theorem",
    "Elaborate on how Raft achieves consensus",
])
def test_reflexion_keywords_route_to_reflexion(question):
    router, _, _, reflexion = _make_router()
    router.chat(question)
    reflexion.chat.assert_called_once()


@pytest.mark.parametrize("question", [
    "What is eventual consistency?",
    "How does a B-tree index work?",
    "What are write-ahead logs?",
    "Summarize chapter 5",
])
def test_default_routes_to_react(question):
    router, react, _, _ = _make_router()
    router.chat(question)
    react.chat.assert_called_once()


def test_plan_keywords_take_priority_over_reflexion():
    """'compare ... in detail' matches both — plan_execute wins."""
    router, _, plan, reflexion = _make_router()
    router.chat("Compare replication and partitioning in detail")
    plan.chat.assert_called_once()
    reflexion.chat.assert_not_called()


def test_router_propagates_last_trace():
    router, react, _, _ = _make_router()
    trace = MagicMock()
    react.last_trace = trace
    router.chat("What is X?")
    assert router.last_trace is trace
