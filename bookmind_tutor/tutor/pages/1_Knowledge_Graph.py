"""
Knowledge Graph visual explorer page.

Displays the user's personal KG (UserEntity nodes + relations) as an interactive
force-directed graph using streamlit-agraph / vis.js. Supports:
  - Full-graph view with text search and entity-type filter
  - Neighborhood explorer: pick a concept and show 1- or 2-hop subgraph
  - Node detail panel: click a node to see its description and connections

Design: plans/kg_visual_page.md
"""
from __future__ import annotations

import os

import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

from dotenv import load_dotenv
load_dotenv()

from bookmind_tutor.knowledge_graph.graph_store import GraphStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "bookmind123")

# Node color + size per entity type
_TYPE_STYLE: dict[str, dict] = {
    "CONCEPT":    {"color": "#4A90D9", "size": 20},
    "SYSTEM":     {"color": "#27AE60", "size": 25},
    "PRINCIPLE":  {"color": "#E67E22", "size": 20},
    "DEFINITION": {"color": "#8E44AD", "size": 18},
    "PERSON":     {"color": "#E74C3C", "size": 18},
}
_DEFAULT_STYLE = {"color": "#95A5A6", "size": 15}

_ALL_TYPES = list(_TYPE_STYLE.keys())


# ---------------------------------------------------------------------------
# Session / init
# ---------------------------------------------------------------------------

def _ensure_graph_store() -> GraphStore:
    """Return the shared GraphStore, creating one if the session lacks it."""
    if "graph_store" not in st.session_state:
        st.session_state.graph_store = GraphStore(
            uri=_NEO4J_URI,
            user=_NEO4J_USER,
            password=_NEO4J_PASSWORD,
        )
    return st.session_state.graph_store


# ---------------------------------------------------------------------------
# Graph building helpers
# ---------------------------------------------------------------------------

def _build_agraph(
    raw_nodes: list[dict],
    raw_edges: list[dict],
    highlight: str = "",
    type_filter: list[str] | None = None,
) -> tuple[list[Node], list[Edge]]:
    """
    Convert raw dicts from GraphStore into streamlit-agraph Node / Edge objects.

    Args:
        highlight: substring to search in node names. Non-matching nodes are
            shown at reduced opacity to keep the graph navigable.
        type_filter: if provided, only nodes of these types are shown.
    """
    active_types = set(type_filter) if type_filter else set(_ALL_TYPES)
    search = highlight.strip().lower()

    # Build node set first so we can filter edges consistently.
    shown_ids: set[str] = set()
    ag_nodes: list[Node] = []

    for n in raw_nodes:
        if n["type"] not in active_types:
            continue
        node_id = n["name"]
        style = _TYPE_STYLE.get(n["type"], _DEFAULT_STYLE)

        # Dim nodes that don't match the search string.
        if search and search not in n["name"].lower():
            color = "#CCCCCC"
            opacity = 0.3
        else:
            color = style["color"]
            opacity = 1.0

        ag_nodes.append(
            Node(
                id=node_id,
                label=node_id,
                size=style["size"],
                color=color,
                # vis.js opacity via font color alpha is limited; we store it
                # in the title tooltip so at least highlighted nodes stand out.
                title=f"[{n['type']}]\n{n['description']}",
                opacity=opacity,
            )
        )
        shown_ids.add(node_id)

    ag_edges: list[Edge] = [
        Edge(
            source=e["from"],
            target=e["to"],
            label=e["type"].replace("_", " ").title(),
            color="#BDC3C7",
        )
        for e in raw_edges
        if e["from"] in shown_ids and e["to"] in shown_ids
    ]

    return ag_nodes, ag_edges


def _build_neighborhood(
    center: str,
    all_nodes: list[dict],
    all_edges: list[dict],
    depth: int,
    graph_store: GraphStore,
) -> tuple[list[Node], list[Edge]]:
    """Return agraph objects for the neighborhood subgraph of `center`."""
    related = graph_store.get_user_related_entities(center, depth=depth)
    neighbor_names = {e.name for e in related} | {center}

    sub_nodes = [n for n in all_nodes if n["name"] in neighbor_names]
    sub_edges = [
        e for e in all_edges
        if e["from"] in neighbor_names and e["to"] in neighbor_names
    ]

    ag_nodes: list[Node] = []
    for n in sub_nodes:
        style = _TYPE_STYLE.get(n["type"], _DEFAULT_STYLE)
        color = "#F39C12" if n["name"] == center else style["color"]
        ag_nodes.append(
            Node(
                id=n["name"],
                label=n["name"],
                size=30 if n["name"] == center else style["size"],
                color=color,
                title=f"[{n['type']}]\n{n['description']}",
            )
        )

    ag_edges = [
        Edge(
            source=e["from"],
            target=e["to"],
            label=e["type"].replace("_", " ").title(),
            color="#BDC3C7",
        )
        for e in sub_edges
    ]

    return ag_nodes, ag_edges


# ---------------------------------------------------------------------------
# Graph config
# ---------------------------------------------------------------------------

def _graph_config(height: int = 520) -> Config:
    return Config(
        width="100%",
        height=height,
        directed=True,
        physics=True,
        hierarchical=False,
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Knowledge Graph - BookMind",
        page_icon="🧠",
        layout="wide",
    )

    graph_store = _ensure_graph_store()
    data = graph_store.get_user_graph_data()
    all_nodes = data["nodes"]
    all_edges = data["edges"]

    st.title("🧠 Knowledge Graph")

    if not all_nodes:
        st.info(
            "Your knowledge graph is empty. "
            "Go to the chat page, ask questions about a book, "
            "and add suggested concepts to start building it."
        )
        st.page_link("app.py", label="Go to chat", icon="💬")
        return

    n_nodes = len(all_nodes)
    n_edges = len(all_edges)
    st.caption(f"{n_nodes} concept(s) · {n_edges} connection(s)")

    # -----------------------------------------------------------------------
    # Controls row
    # -----------------------------------------------------------------------
    col_search, col_type, col_mode = st.columns([3, 2, 2])

    with col_search:
        search_text = st.text_input(
            "Search concepts",
            placeholder="Filter by name...",
            label_visibility="collapsed",
        )

    with col_type:
        present_types = sorted({n["type"] for n in all_nodes})
        type_filter = st.multiselect(
            "Entity type",
            options=present_types,
            default=present_types,
            label_visibility="collapsed",
            placeholder="Filter by type...",
        )

    with col_mode:
        view_mode = st.radio(
            "View",
            options=["Full graph", "Neighborhood"],
            horizontal=True,
            label_visibility="collapsed",
        )

    st.divider()

    # -----------------------------------------------------------------------
    # Graph + detail panel
    # -----------------------------------------------------------------------
    graph_col, detail_col = st.columns([3, 1])

    with graph_col:
        if view_mode == "Full graph":
            ag_nodes, ag_edges = _build_agraph(
                all_nodes, all_edges,
                highlight=search_text,
                type_filter=type_filter or None,
            )
        else:
            # Neighborhood mode
            node_names = sorted(n["name"] for n in all_nodes)
            center = st.selectbox(
                "Center concept",
                options=node_names,
                index=0,
            )
            depth = st.radio("Depth", options=[1, 2], horizontal=True)
            ag_nodes, ag_edges = _build_neighborhood(
                center, all_nodes, all_edges, depth, graph_store
            )

        if not ag_nodes:
            st.warning("No nodes match your current filters.")
            clicked_node = None
        else:
            clicked_node = agraph(
                nodes=ag_nodes,
                edges=ag_edges,
                config=_graph_config(),
            )

    with detail_col:
        st.subheader("Node detail")
        if clicked_node:
            node_data = next(
                (n for n in all_nodes if n["name"] == clicked_node), None
            )
            if node_data:
                style = _TYPE_STYLE.get(node_data["type"], _DEFAULT_STYLE)
                st.markdown(
                    f"**{node_data['name']}**  \n"
                    f"<span style='color:{style['color']};font-weight:bold'>"
                    f"{node_data['type']}</span>",
                    unsafe_allow_html=True,
                )
                if node_data["description"]:
                    st.write(node_data["description"])
                else:
                    st.caption("No description available.")

                related = graph_store.get_user_related_entities(clicked_node, depth=1)
                if related:
                    st.markdown("**Connected to:**")
                    for r in related:
                        st.write(f"- {r.name} ({r.type})")
                else:
                    st.caption("No connections yet.")

                st.divider()
                if st.button(
                    "Remove from my KG",
                    key=f"remove_{clicked_node}",
                    type="secondary",
                ):
                    graph_store.delete_user_entity(clicked_node)
                    st.rerun()
        else:
            st.caption("Click a node in the graph to see its details here.")

        st.divider()

        # Legend
        st.markdown("**Legend**")
        for t, s in _TYPE_STYLE.items():
            if t in {n["type"] for n in all_nodes}:
                st.markdown(
                    f"<span style='color:{s['color']}'>●</span> {t}",
                    unsafe_allow_html=True,
                )

    # -----------------------------------------------------------------------
    # Footer: navigation + danger zone
    # -----------------------------------------------------------------------
    st.divider()
    footer_left, footer_right = st.columns([3, 1])
    with footer_left:
        st.page_link("app.py", label="Back to chat", icon="💬")
    with footer_right:
        if st.button("Clear all", type="secondary", use_container_width=True):
            graph_store.clear_user_graph()
            st.rerun()


if __name__ == "__main__":
    main()
