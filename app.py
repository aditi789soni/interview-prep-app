"""
AI Technical & Behavioral Interviewer
Single-file Streamlit application featuring:
1. Multi-role selection (DSA, System Design, Behavioral)
2. Structured Question Generation with Pydantic & Google Gemini API (google-genai)
3. 5-Point Rubric Answer Evaluation (Communication, Correctness, Optimization)
4. Local Vector Retrieval / RAG setup with pre-seeded question banks
5. Environment variable & Sidebar GEMINI_API_KEY configuration
"""

import os
import json
import math
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

import streamlit as st
from pydantic import BaseModel, Field

# Load .env file automatically
try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv("../.env")
except ImportError:
    pass


# ---------------------------------------------------------
# Google GenAI Imports with Graceful Fallback Detection
# ---------------------------------------------------------
GENAI_AVAILABLE = False
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    GENAI_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------
# Pydantic Schemas for Structured JSON Output
# ---------------------------------------------------------
class GeneratedQuestion(BaseModel):
    title: str = Field(description="Concise title of the interview question")
    role: str = Field(description="Interview track: DSA, System Design, or Behavioral")
    difficulty: str = Field(description="Target difficulty level, e.g., Junior, Mid-Level, Senior, Staff")
    problem_statement: str = Field(description="Comprehensive interview question prompt or problem scenario")
    constraints_or_context: List[str] = Field(description="List of technical constraints, edge cases, or scenario context")
    hints: List[str] = Field(description="Progressive interview hints to offer if the candidate gets stuck")
    evaluation_criteria: List[str] = Field(description="Specific milestones or topics an interviewer expects to hear")


class RubricCriterion(BaseModel):
    score: int = Field(ge=1, le=5, description="Numerical score from 1 (Unacceptable/Poor) to 5 (Exemplary/Mastery)")
    rating_label: str = Field(description="Label: Poor, Below Average, Competent, Strong, or Exemplary")
    feedback: str = Field(description="Constructive critique detailing rationale for this specific score")


class EvaluationReport(BaseModel):
    overall_score: float = Field(description="Overall weighted average score out of 5.0")
    verdict: str = Field(description="Hiring recommendation: Strong No Hire, Lean No Hire, Lean Hire, Strong Hire")
    communication: RubricCriterion = Field(description="Clarity, structure, STAR method, trade-off articulation")
    correctness: RubricCriterion = Field(description="Technical accuracy, algorithmic validity, system resilience, truthfulness")
    optimization: RubricCriterion = Field(description="Time/space complexity, scalability, edge case handling, efficiency")
    strengths: List[str] = Field(description="Key strengths demonstrated in the response")
    areas_for_improvement: List[str] = Field(description="Concrete actionable gaps or missed optimizations")
    ideal_response_outline: str = Field(description="High-level blueprint of what an ideal 5/5 answer includes")


# ---------------------------------------------------------
# Vector Store / RAG Engine (Local Chroma-Compatible Mock)
# ---------------------------------------------------------
class LocalVectorStore:
    """
    Lightweight, self-contained Vector Store implementation.
    Features:
    - Pre-seeded domain questions for DSA, System Design, and Behavioral
    - Cosine similarity search over normalized term-frequency embeddings
    - Optional Gemini embedding integration when API key is present
    - Metadata filtering by role and difficulty
    """
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self._seed_sample_bank()

    def _seed_sample_bank(self):
        sample_questions = [
            # DSA Questions
            {
                "id": "dsa-1",
                "role": "DSA",
                "difficulty": "Medium",
                "title": "LRU Cache Implementation",
                "text": "Design and implement a data structure for Least Recently Used (LRU) cache. "
                        "Support get(key) and put(key, value) operations both in O(1) average time complexity. "
                        "Requires Doubly Linked List and Hash Map hash table synchronization.",
                "tags": ["hash-table", "linked-list", "design", "o(1)"],
            },
            {
                "id": "dsa-2",
                "role": "DSA",
                "difficulty": "Hard",
                "title": "Trapping Rain Water",
                "text": "Given n non-negative integers representing an elevation map where the width of each bar is 1, "
                        "compute how much water it can trap after raining. "
                        "Optimal solution uses two pointers with O(n) time and O(1) auxiliary space.",
                "tags": ["two-pointers", "arrays", "stack", "dynamic-programming"],
            },
            {
                "id": "dsa-3",
                "role": "DSA",
                "difficulty": "Medium",
                "title": "Course Schedule & Cycle Detection",
                "text": "Determine if all courses can be finished given prerequisite pairs. "
                        "Models graph topological sorting and cycle detection using Kahn's algorithm (BFS with in-degrees) "
                        "or Depth-First Search with recursion call stack coloring (white/gray/black).",
                "tags": ["graphs", "bfs", "dfs", "topological-sort"],
            },
            {
                "id": "dsa-4",
                "role": "DSA",
                "difficulty": "Hard",
                "title": "Median of Two Sorted Arrays",
                "text": "Given two sorted arrays nums1 and nums2 of size m and n, return the median of the two sorted arrays. "
                        "The overall run time complexity should be O(log(min(m, n))) using binary search on partition boundaries.",
                "tags": ["binary-search", "arrays", "divide-and-conquer"],
            },
            {
                "id": "dsa-5",
                "role": "DSA",
                "difficulty": "Easy",
                "title": "Valid Anagram and Substring Permutations",
                "text": "Determine if string t is an anagram of s. Check character frequency maps or fixed size ASCII buckets. "
                        "Extend to sliding window pattern for finding all anagram substrings in O(n) time.",
                "tags": ["hash-table", "string", "frequency-count"],
            },

            # System Design Questions
            {
                "id": "sys-1",
                "role": "System Design",
                "difficulty": "Senior",
                "title": "Scalable URL Shortener (TinyURL)",
                "text": "Design a globally distributed URL shortener service handling 100M new URLs per day with 10:1 read-to-write ratio. "
                        "Discuss Base62 encoding, distributed ID generation (Snowflake/Ticket Service), caching with Redis, "
                        "database partitioning (sharding by hash), and 301 vs 302 redirection trade-offs.",
                "tags": ["distributed-systems", "hashing", "caching", "db-sharding"],
            },
            {
                "id": "sys-2",
                "role": "System Design",
                "difficulty": "Senior",
                "title": "Distributed Rate Limiter",
                "text": "Design an API rate limiter service enforcing tier-based rate limits across microservices with sub-millisecond latency. "
                        "Compare Token Bucket, Leaky Bucket, and Sliding Window Log algorithms. Address race conditions using Redis Lua scripts "
                        "and multi-datacenter synchronization strategies.",
                "tags": ["rate-limiting", "redis", "concurrency", "fault-tolerance"],
            },
            {
                "id": "sys-3",
                "role": "System Design",
                "difficulty": "Staff",
                "title": "Real-Time Collaborative Document Editor",
                "text": "Architect Google Docs-style real-time collaborative text editing with offline edits and peer synchronization. "
                        "Deep-dive into Conflict-free Replicated Data Types (CRDTs) vs Operational Transformation (OT), "
                        "WebSocket connection multiplexing, message ordering with vector clocks, and snapshot persistence in object storage.",
                "tags": ["crdt", "websockets", "consistency", "event-driven"],
            },
            {
                "id": "sys-4",
                "role": "System Design",
                "difficulty": "Mid-Level",
                "title": "Push Notification Delivery Pipeline",
                "text": "Design a high-volume notification engine dispatching millions of messages across APNS, FCM, SMS, and Email. "
                        "Address queueing with Kafka/RabbitMQ, message deduplication with idempotency keys, rate throttling to vendor APIs, "
                        "and user notification preference filtering.",
                "tags": ["queues", "kafka", "idempotency", "resilience"],
            },

            # Behavioral Questions
            {
                "id": "beh-1",
                "role": "Behavioral",
                "difficulty": "All Levels",
                "title": "Technical Disagreement & Influence",
                "text": "Tell me about a time you strongly disagreed with a tech lead or architectural decision. "
                        "Evaluates emotional intelligence, data-driven prototyping, respectful communication, "
                        "and the ability to disagree and commit once a decision is finalized (STAR method).",
                "tags": ["conflict-resolution", "leadership", "star-method"],
            },
            {
                "id": "beh-2",
                "role": "Behavioral",
                "difficulty": "Senior",
                "title": "High-Severity Production Incident Response",
                "text": "Describe a critical production outage or data regression you diagnosed and resolved under high stakes. "
                        "Examines triage under pressure, blameless post-mortem culture, root cause analysis (5 Whys), "
                        "and instituting long-term systemic safeguards.",
                "tags": ["incident-management", "ownership", "post-mortem"],
            },
            {
                "id": "beh-3",
                "role": "Behavioral",
                "difficulty": "Mid-Level",
                "title": "Project Failure and Pivoting",
                "text": "Describe a project that fell behind schedule, missed key OKRs, or ultimately failed. "
                        "Assesses intellectual honesty, accountability without finger-pointing, constructive takeaways, "
                        "and how learning changed your future engineering practices.",
                "tags": ["accountability", "growth-mindset", "resilience"],
            },
            {
                "id": "beh-4",
                "role": "Behavioral",
                "difficulty": "Staff",
                "title": "Navigating Ambiguity and Cross-Team Alignment",
                "text": "Tell me about leading a vaguely defined strategic initiative requiring cross-functional consensus across Product, Eng, and Ops. "
                        "Evaluates roadmap framing, stakeholder alignment, de-risking milestones, and unblocking team dependencies.",
                "tags": ["strategic-leadership", "cross-functional", "ambiguity"],
            },
        ]
        self.documents = sample_questions

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
        stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "you", "are", "have", "has", "can",
            "not", "all", "any", "how", "what", "when", "using", "your", "into", "over", "such"
        }
        return [w for w in words if w not in stopwords]

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        tf: Dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0.0) + 1.0
        # Normalize vector magnitude
        norm = math.sqrt(sum(v * v for v in tf.values())) or 1.0
        return {k: v / norm for k, v in tf.items()}

    def _cosine_similarity(self, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        score = 0.0
        for token, val in vec_a.items():
            if token in vec_b:
                score += val * vec_b[token]
        return score

    def add_document(self, doc_id: str, role: str, difficulty: str, title: str, text: str, tags: List[str]):
        self.documents.append({
            "id": doc_id,
            "role": role,
            "difficulty": difficulty,
            "title": title,
            "text": text,
            "tags": tags,
        })

    def query(self, query_text: str, role_filter: Optional[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        query_vec = self._vectorize(self._tokenize(query_text))
        results = []

        for doc in self.documents:
            if role_filter and doc["role"].lower() != role_filter.lower():
                continue

            doc_text = f"{doc['title']} {doc['text']} {' '.join(doc.get('tags', []))}"
            doc_vec = self._vectorize(self._tokenize(doc_text))
            sim = self._cosine_similarity(query_vec, doc_vec)

            # Boost if exact tag match
            query_tokens = set(self._tokenize(query_text))
            tag_overlap = sum(1 for tag in doc.get("tags", []) if tag.lower() in query_tokens)
            boosted_sim = min(1.0, sim + (0.15 * tag_overlap))

            results.append({
                "document": doc,
                "score": round(boosted_sim, 3)
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# ---------------------------------------------------------
# Streamlit Application State Initialization
# ---------------------------------------------------------
st.set_page_config(
    page_title="Gemini Mock Interviewer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = LocalVectorStore()

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "retrieved_rag_context" not in st.session_state:
    st.session_state.retrieved_rag_context = []

if "evaluation_result" not in st.session_state:
    st.session_state.evaluation_result = None

if "evaluation_history" not in st.session_state:
    st.session_state.evaluation_history = []


# ---------------------------------------------------------
# Sidebar: API Key, Model Selection & Status
# ---------------------------------------------------------
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/streamlit/brand/master/streamlit-mark-color.png",
        width=48,
    )
    st.title("Interview AI Control")
    st.caption("Powered by Google Gemini & Pydantic Structured Outputs")

    # API Key retrieval logic
    env_api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if env_api_key:
        st.success("✅ GEMINI_API_KEY found in Environment", icon="🔒")
        api_key_input = st.text_input(
            "Override API Key (Optional)",
            value=env_api_key,
            type="password",
            help="Pre-populated from GEMINI_API_KEY environment variable.",
        )
    else:
        st.warning("⚠️ No API Key in Environment", icon="🔑")
        api_key_input = st.text_input(
            "Enter Gemini API Key",
            type="password",
            help="Get your free key at https://aistudio.google.com/app/apikey",
        )

    active_api_key = api_key_input.strip() or env_api_key

    st.markdown("---")
    st.subheader("Model Configuration")

    model_options = [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-2.5-flash (legacy alias)",
    ]
    selected_model_raw = st.selectbox(
        "Select Gemini Model",
        options=model_options,
        index=0,
        help="gemini-3.8-flash provides fast, high-quality reasoning and structured outputs.",
    )
    # Map selection to canonical model
    selected_model = "gemini-3.8-flash" if "2.5-flash" in selected_model_raw else selected_model_raw

    st.markdown("---")
    st.subheader("RAG Vector Store Status")
    doc_count = len(st.session_state.vector_store.documents)
    st.metric(label="Indexed Interview Questions", value=doc_count)

    with st.expander("➕ Add Custom RAG Question"):
        with st.form("add_question_form"):
            new_role = st.selectbox("Role", ["DSA", "System Design", "Behavioral"])
            new_diff = st.selectbox("Difficulty", ["Easy", "Medium", "Hard", "Senior", "Staff"])
            new_title = st.text_input("Title", placeholder="e.g. Distributed Cache Replication")
            new_text = st.text_area("Question & Rubric Context", placeholder="Describe problem details, key trade-offs...")
            new_tags = st.text_input("Tags (comma separated)", placeholder="distributed, consensus, raft")
            submitted = st.form_submit_button("Index to Vector Store")
            if submitted and new_title and new_text:
                st.session_state.vector_store.add_document(
                    doc_id=f"custom-{datetime.now().strftime('%H%M%S')}",
                    role=new_role,
                    difficulty=new_diff,
                    title=new_title,
                    text=new_text,
                    tags=[t.strip() for t in new_tags.split(",") if t.strip()]
                )
                st.success("Successfully added to local vector store!")
                st.rerun()

    st.markdown("---")
    st.caption("Built with Streamlit • google-genai • Pydantic v2")


# ---------------------------------------------------------
# Helper: Gemini Client Factory & Structured Output Caller
# ---------------------------------------------------------
def get_genai_client(api_key: str):
    if not GENAI_AVAILABLE:
        raise RuntimeError("The `google-genai` package is not installed. Please install it with `pip install google-genai`.")
    if not api_key:
        raise ValueError("Missing Gemini API Key. Please provide one in the sidebar or set the GEMINI_API_KEY environment variable.")
    return genai.Client(api_key=api_key)


def generate_structured_gemini_response(
    prompt: str,
    response_model: type[BaseModel],
    api_key: str,
    model_name: str = "gemini-3.8-flash"
) -> BaseModel:
    """
    Invokes Gemini API with enforced Pydantic structured output.
    Uses google-genai Client with types.GenerateContentConfig.
    """
    client = get_genai_client(api_key)

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_model,
        temperature=0.4,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )

    if not response.text:
        raise ValueError("Empty response received from Gemini API.")

    return response_model.model_validate_json(response.text)


# ---------------------------------------------------------
# Mock Response Fallback (Demo Mode when API key not yet set)
# ---------------------------------------------------------
def get_mock_generated_question(role: str, difficulty: str, topic: str) -> GeneratedQuestion:
    if role == "DSA":
        return GeneratedQuestion(
            title=f"Optimal Subarray Sum with Dynamic Bounds ({topic or 'Sliding Window'})",
            role="DSA",
            difficulty=difficulty,
            problem_statement=(
                f"Given an integer array nums and integers k and limit, find the maximum subarray sum "
                f"such that the length of the subarray is at least k and no adjacent elements differ by more than limit. "
                f"Focus on achieving O(N) or O(N log N) time complexity."
            ),
            constraints_or_context=[
                "1 <= nums.length <= 10^5",
                "-10^4 <= nums[i] <= 10^4",
                "1 <= k <= nums.length",
                "Time limit: 2.0s in Python / 1.0s in C++",
            ],
            hints=[
                "Consider whether a monotonic deque or sliding window can maintain min/max in range.",
                "Can prefix sums help evaluate subarray totals in O(1) once boundary validity is ensured?",
                "Think about segment trees or two pointers if dynamic constraint violations occur.",
            ],
            evaluation_criteria=[
                "Handles negative numbers correctly without assuming greedy positivity.",
                "Explains time and space complexity tradeoffs explicitly.",
                "Thinks out loud about edge cases like all-negative arrays or k equal to array length.",
            ]
        )
    elif role == "System Design":
        return GeneratedQuestion(
            title=f"Global Real-Time Leaderboard System ({topic or 'High Concurrency'})",
            role="System Design",
            difficulty=difficulty,
            problem_statement=(
                f"Design a real-time gaming leaderboard serving 50 million monthly active users, "
                f"with 100,000 score submissions per second at peak. Users can query their exact global rank "
                f"and view top 100 players with sub-50ms latency."
            ),
            constraints_or_context=[
                "50M MAU, 5M DAU, 100k writes/sec peak.",
                "Read-to-write ratio is approximately 3:1.",
                "Global rank lookups must be strictly accurate or bounded within 1s staleness.",
                "High availability across 3 geographic regions.",
            ],
            hints=[
                "Examine Redis Sorted Sets (ZADD, ZREVRANGE, ZREVRANK) and memory sharding limits.",
                "How does write sharding or tiered skip-lists prevent a single Redis node bottleneck?",
                "Consider event-driven architecture using Kafka for durable score ingestion.",
            ],
            evaluation_criteria=[
                "Capacity estimation (QPS, memory footprint for 50M members in sorted sets).",
                "Partitioning strategy and caching layer design.",
                "Resilience during network partition or Redis primary node failover.",
            ]
        )
    else:
        return GeneratedQuestion(
            title=f"Navigating Critical Project Deadlines & Cross-Team Friction ({topic or 'Leadership'})",
            role="Behavioral",
            difficulty=difficulty,
            problem_statement=(
                f"Describe a situation where a critical feature delivery was jeopardized due to an upstream dependency "
                f"missing their SLA or proposing an incompatible API contract. How did you handle the situation, "
                f"align leadership, and deliver without damaging working relationships?"
            ),
            constraints_or_context=[
                "Use the STAR methodology (Situation, Task, Action, Result).",
                "Emphasize personal agency and active problem solving rather than passive escalation.",
                "Quantify the business impact and ongoing team improvements.",
            ],
            hints=[
                "Did you create a fallback adapter / feature flag to decouple releases?",
                "How did you address the human dynamic without assigning blame in public forums?",
                "What metrics or post-launch retrospectives were established?",
            ],
            evaluation_criteria=[
                "Clear demonstration of the STAR framework structure.",
                "Ownership mindset and emotional maturity in resolving conflicting priorities.",
                "Measurable outcome and long-term systemic process improvement.",
            ]
        )


def get_mock_evaluation(question_title: str, user_answer: str) -> EvaluationReport:
    word_count = len(user_answer.split())
    base_score = 4 if word_count > 60 else (3 if word_count > 25 else 2)

    return EvaluationReport(
        overall_score=round(base_score + 0.3, 1),
        verdict="Lean Hire" if base_score >= 4 else "Lean No Hire",
        communication=RubricCriterion(
            score=base_score,
            rating_label="Strong" if base_score >= 4 else "Competent",
            feedback=(
                "Your thoughts were logically structured and easy to follow. You articulated your assumptions upfront. "
                "To reach a 5/5, clearly delineate each phase using explicit headers or STAR signposting."
            )
        ),
        correctness=RubricCriterion(
            score=base_score,
            rating_label="Strong" if base_score >= 4 else "Competent",
            feedback=(
                "Technical solution correctly addresses the core requirements and constraints. "
                "Be sure to explicitly verify corner cases (e.g. empty inputs, zero division, or partition failures)."
            )
        ),
        optimization=RubricCriterion(
            score=max(1, base_score - 1),
            rating_label="Competent" if base_score >= 4 else "Below Average",
            feedback=(
                "You identified the primary algorithmic/architectural path, but missed mentioning secondary bottlenecks "
                "such as memory overhead, cache eviction behavior, or I/O contention under load."
            )
        ),
        strengths=[
            "Direct and relevant answer that attacks the heart of the problem statement.",
            "Demonstrated solid familiarity with core engineering idioms and principles.",
            "Recognized the primary trade-offs without over-engineering prematurely."
        ],
        areas_for_improvement=[
            "State time and space complexities explicitly before writing implementation details.",
            "Proactively discuss monitoring, alerting, telemetry, or regression tests.",
            "Provide quantitative benchmarks or concrete scale numbers when justifying choices."
        ],
        ideal_response_outline=(
            f"An optimal response to '{question_title}' begins with clarifying questions and boundary definitions, "
            f"proposes a clean baseline solution with complexity analysis, explores trade-offs (e.g., memory vs compute), "
            f"and concludes with failure recovery, unit test validation, and scalability observations."
        )
    )


# ---------------------------------------------------------
# Main UI Layout & Tabs
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 12px;
    }
    .score-badge {
        font-size: 1.15rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-block;
    }
    .tag-chip {
        background-color: #e9ecef;
        color: #495057;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-right: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎯 AI Technical Interview Simulator")
st.markdown(
    "Practice coding challenges, distributed system designs, and behavioral scenarios with real-time "
    "**Gemini structured reasoning**, **local RAG retrieval**, and **5-point rubric scoring**."
)

tab_roles, tab_generate, tab_evaluate = st.tabs([
    "1️⃣ Role Selection & RAG Bank",
    "2️⃣ Question Generation",
    "3️⃣ Answer Evaluation & Rubric"
])


# =========================================================
# TAB 1: Role Selection & RAG Question Bank
# =========================================================
with tab_roles:
    st.header("Select Interview Track")

    col_role1, col_role2, col_role3 = st.columns(3)

    with col_role1:
        st.subheader("💻 DSA Track")
        st.markdown(
            "Algorithmic challenges, time & space complexity, dynamic programming, "
            "graphs, trees, and optimal data structure manipulation."
        )
        if st.button("Select Data Structures & Algorithms", use_container_width=True):
            st.session_state.selected_track = "DSA"
            st.toast("Selected DSA track!", icon="💻")

    with col_role2:
        st.subheader("🏛️ System Design Track")
        st.markdown(
            "Scalable distributed systems, microservices, databases, caching, "
            "rate limiting, consensus protocols, and fault tolerance."
        )
        if st.button("Select System Design", use_container_width=True):
            st.session_state.selected_track = "System Design"
            st.toast("Selected System Design track!", icon="🏛️")

    with col_role3:
        st.subheader("🤝 Behavioral Track")
        st.markdown(
            "Leadership principles, conflict resolution, technical trade-offs, "
            "STAR methodology, project retrospectives, and engineering culture."
        )
        if st.button("Select Behavioral", use_container_width=True):
            st.session_state.selected_track = "Behavioral"
            st.toast("Selected Behavioral track!", icon="🤝")

    current_track = st.session_state.get("selected_track", "DSA")
    st.info(f"👉 **Currently Selected Track:** `{current_track}`", icon="🎯")

    st.markdown("---")
    st.subheader(f"📚 Local RAG Question Bank Preview ({current_track})")
    st.caption("These pre-indexed problems are retrieved by the RAG module to ground the Gemini question generator.")

    filtered_docs = [
        d for d in st.session_state.vector_store.documents
        if d["role"].lower() == current_track.lower()
    ]

    for doc in filtered_docs:
        with st.expander(f"📌 {doc['title']} • [{doc['difficulty']}]"):
            st.write(doc["text"])
            tags_html = " ".join([f'<span class="tag-chip">{t}</span>' for t in doc.get("tags", [])])
            st.markdown(tags_html, unsafe_allow_html=True)


# =========================================================
# TAB 2: Question Generation (RAG + Gemini Structured Output)
# =========================================================
with tab_generate:
    st.header("Generate Tailored Interview Question")

    current_track = st.session_state.get("selected_track", "DSA")

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        difficulty_level = st.selectbox(
            "Seniority Level",
            options=["Junior / Entry-Level", "Mid-Level", "Senior Engineer", "Staff / Principal"],
            index=1
        )
    with col_cfg2:
        topic_focus = st.text_input(
            "Topic / Focus Keyword",
            value="",
            placeholder="e.g., Sliding Window, Redis Caching, Conflict Resolution",
            help="Used to query the local vector store for relevant problem context."
        )
    with col_cfg3:
        rag_top_k = st.slider("RAG Context Depth (Top-K)", min_value=1, max_value=4, value=2)

    gen_button = st.button("🚀 Generate Structured Question with Gemini", type="primary", use_container_width=True)

    if gen_button:
        with st.spinner("Retrieving RAG vectors and querying Gemini..."):
            # 1. RAG Vector Retrieval
            query_str = f"{current_track} {difficulty_level} {topic_focus}"
            rag_matches = st.session_state.vector_store.query(
                query_text=query_str,
                role_filter=current_track,
                top_k=rag_top_k
            )
            st.session_state.retrieved_rag_context = rag_matches

            # 2. Build RAG prompt context
            context_snippets = []
            for idx, match in enumerate(rag_matches, 1):
                doc = match["document"]
                sim = match["score"]
                context_snippets.append(
                    f"[Sample {idx} (Similarity {sim:.2f})]: Title: {doc['title']} | Content: {doc['text']}"
                )
            context_block = "\n".join(context_snippets)

            system_and_user_prompt = f"""
You are an expert principal technical interviewer at a top-tier technology company.
Generate an authentic, challenging interview question for the track: '{current_track}'.

Seniority Level: {difficulty_level}
Focus Subject: {topic_focus if topic_focus else 'General Core Competency'}

Use the following retrieved reference questions from our knowledge base for tone and benchmark calibration:
--- RETRIEVED RAG CONTEXT ---
{context_block}
-----------------------------

Create a unique, comprehensive problem tailored to this role.
Ensure the problem statement has clear expectations, realistic constraints, progressive hints, and precise evaluation criteria.
"""

            # 3. Call Gemini with Structured Pydantic Output
            if active_api_key:
                try:
                    generated_q = generate_structured_gemini_response(
                        prompt=system_and_user_prompt,
                        response_model=GeneratedQuestion,
                        api_key=active_api_key,
                        model_name=selected_model
                    )
                    st.session_state.current_question = generated_q
                    st.success("Successfully generated question using Gemini API!")
                except Exception as err:
                    st.error(f"Gemini API Error: {err}. Falling back to high-fidelity mock generator.")
                    st.session_state.current_question = get_mock_generated_question(current_track, difficulty_level, topic_focus)
            else:
                st.info("💡 No GEMINI_API_KEY provided; generating using realistic offline mock generator.")
                st.session_state.current_question = get_mock_generated_question(current_track, difficulty_level, topic_focus)

    # Display RAG Insights
    if st.session_state.retrieved_rag_context:
        with st.expander("🔍 Inspect Retrieved RAG Context & Similarity Scores", expanded=False):
            st.write(f"Query matched {len(st.session_state.retrieved_rag_context)} documents from vector space:")
            for m in st.session_state.retrieved_rag_context:
                d = m["document"]
                st.markdown(
                    f"- **{d['title']}** (Score: `{m['score']}` | {d['difficulty']}): *{d['text'][:120]}...*"
                )

    # Display Generated Question
    q: Optional[GeneratedQuestion] = st.session_state.current_question
    if q:
        st.markdown("---")
        col_q_title, col_q_badge = st.columns([3, 1])
        with col_q_title:
            st.subheader(f"📋 {q.title}")
        with col_q_badge:
            st.markdown(f"**Track:** `{q.role}` | **Level:** `{q.difficulty}`")

        st.markdown("### Problem Statement")
        st.write(q.problem_statement)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### ⚙️ Constraints & System Context")
            for c in q.constraints_or_context:
                st.markdown(f"- {c}")

            st.markdown("#### 🎯 Evaluation Criteria (Interview Milestones)")
            for crit in q.evaluation_criteria:
                st.markdown(f"- {crit}")

        with col_right:
            st.markdown("#### 💡 Progressive Hints")
            for idx, h in enumerate(q.hints, 1):
                with st.expander(f"Hint #{idx}"):
                    st.write(h)

        st.success("Ready for evaluation! Head over to the **3️⃣ Answer Evaluation & Rubric** tab to submit your response.", icon="🚀")


# =========================================================
# TAB 3: Answer Evaluation & 5-Point Rubric
# =========================================================
with tab_evaluate:
    st.header("Answer Submission & 5-Point Rubric Evaluation")

    active_q = st.session_state.current_question

    if not active_q:
        st.warning("⚠️ No active question found. Please generate or select a question in Tab 2 first.")
    else:
        st.markdown(f"**Target Question:** `{active_q.title}` (`{active_q.role}` - `{active_q.difficulty}`)")
        with st.expander("Show Question Summary"):
            st.write(active_q.problem_statement)

        user_code_or_answer = st.text_area(
            "Enter your solution, code, architectural diagram notes, or STAR behavioral narrative:",
            height=240,
            placeholder="Type your complete solution here...\n- For DSA: Code implementation, algorithm walkthrough, time & space complexities.\n- For System Design: High-level design, components, database schema, scaling.\n- For Behavioral: Situation, Task, Action, and quantifiable Result."
        )

        col_eval_btn, col_sample_btn = st.columns([2, 1])
        with col_eval_btn:
            eval_clicked = st.button("⚖️ Evaluate Answer (5-Point Rubric)", type="primary", use_container_width=True)
        with col_sample_btn:
            load_sample = st.button("📝 Load Sample Answer", use_container_width=True)

        if load_sample:
            sample_answers = {
                "DSA": (
                    "def maxSubarraySum(nums, k, limit):\n"
                    "    # Sliding window with deque for min/max within window\n"
                    "    from collections import deque\n"
                    "    prefix = [0]\n"
                    "    for x in nums:\n"
                    "        prefix.append(prefix[-1] + x)\n"
                    "    max_sum = float('-inf')\n"
                    "    # Check validity of adjacent elements\n"
                    "    for i in range(len(nums) - k + 1):\n"
                    "        curr_sum = prefix[i+k] - prefix[i]\n"
                    "        max_sum = max(max_sum, curr_sum)\n"
                    "    # Time Complexity: O(N) using sliding window\n"
                    "    # Space Complexity: O(N) for prefix sum\n"
                    "    return max_sum"
                ),
                "System Design": (
                    "1. Scale: 100k writes/s peak, 50M MAU.\n"
                    "2. Storage & Memory: Redis Cluster with Sorted Sets (ZADD, ZREVRANK).\n"
                    "   50M users * 32 bytes/entry ≈ 1.6 GB RAM, easily fit in memory.\n"
                    "3. Write Pipeline: API Gateway -> Kafka -> Leaderboard Worker Service -> Redis.\n"
                    "4. Read Path: CDN / Local Cache for Top 100 (TTL 1s), Direct Redis read for personal rank.\n"
                    "5. Sharding: Tier-based or country-based sharding to eliminate hot Redis master bottlenecks."
                ),
                "Behavioral": (
                    "Situation: In Q3, our team was migrating our auth service when a dependent team changed the token format 3 days before code freeze.\n"
                    "Task: As tech lead, I needed to avoid slipping our release date without burning out the team.\n"
                    "Action: I scheduled an emergency alignment sync, proposed an adapter shim that supported both legacy and new token specs, and authored the adapter within 4 hours. We feature-flagged the rollout.\n"
                    "Result: Zero downtime, launched on schedule, and instituted weekly API contract reviews that prevented subsequent regressions."
                )
            }
            user_code_or_answer = sample_answers.get(active_q.role, sample_answers["DSA"])
            st.rerun()

        if eval_clicked:
            if not user_code_or_answer.strip():
                st.error("Please enter a solution before requesting evaluation.")
            else:
                with st.spinner("Scoring response across Communication, Correctness, and Optimization..."):
                    eval_prompt = f"""
You are an expert calibration interviewer evaluating a candidate's answer.
QUESTION TITLE: {active_q.title}
TRACK: {active_q.role}
SENIORITY: {active_q.difficulty}
PROBLEM SPECIFICATION:
{active_q.problem_statement}

CONSTRAINTS:
{json.dumps(active_q.constraints_or_context)}

EVALUATION CRITERIA:
{json.dumps(active_q.evaluation_criteria)}

CANDIDATE'S SUBMITTED RESPONSE:
\"\"\"
{user_code_or_answer}
\"\"\"

Rubric Guidelines (Scores 1 to 5):
1 - Poor / Inadequate: Major flaws, inaccurate, or missing core principles.
2 - Below Average: Partial understanding, significant omissions or unhandled edge cases.
3 - Competent: Meets baseline expectations, sound reasoning, minor gaps.
4 - Strong: Thorough, clean, clear trade-offs, optimal complexity or STAR structure.
5 - Exemplary: Flawless mastery, proactive edge case handling, optimal scale reasoning, crisp communication.

Score the candidate across:
1. Communication (Clarity, structure, STAR method or code readability, articulating assumptions)
2. Correctness (Technical correctness, bug-free, algorithm validity, truthful system constraints)
3. Optimization (Time/space complexity, performance, scalability, bottlenecks, edge case coverage)

Provide hiring recommendation, strengths, areas for improvement, and ideal response outline.
"""
                    if active_api_key:
                        try:
                            eval_rep = generate_structured_gemini_response(
                                prompt=eval_prompt,
                                response_model=EvaluationReport,
                                api_key=active_api_key,
                                model_name=selected_model
                            )
                            st.session_state.evaluation_result = eval_rep
                            st.session_state.evaluation_history.append({
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "question": active_q.title,
                                "role": active_q.role,
                                "overall_score": eval_rep.overall_score,
                                "verdict": eval_rep.verdict
                            })
                            st.success("Evaluation completed with Gemini API!")
                        except Exception as e:
                            st.error(f"Gemini API Error: {e}. Generating offline rubric evaluation.")
                            st.session_state.evaluation_result = get_mock_evaluation(active_q.title, user_code_or_answer)
                    else:
                        st.info("💡 No GEMINI_API_KEY provided; providing offline rubric evaluation.")
                        st.session_state.evaluation_result = get_mock_evaluation(active_q.title, user_code_or_answer)

    # Render Evaluation Report
    report: Optional[EvaluationReport] = st.session_state.evaluation_result
    if report:
        st.markdown("---")
        st.subheader("📊 Candidate Evaluation Scorecard")

        # Top summary metrics
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Overall Score", f"{report.overall_score:.1f} / 5.0")
        with m_col2:
            st.metric("Communication", f"{report.communication.score} / 5")
        with m_col3:
            st.metric("Correctness", f"{report.correctness.score} / 5")
        with m_col4:
            st.metric("Optimization", f"{report.optimization.score} / 5")

        verdict_color = "green" if "Hire" in report.verdict and "No" not in report.verdict else "orange"
        st.markdown(f"**Hiring Recommendation:** :{verdict_color}[**{report.verdict}**]")

        st.markdown("### 📋 5-Point Rubric Breakdown")

        r_col1, r_col2, r_col3 = st.columns(3)

        with r_col1:
            st.markdown(f"#### 🗣️ Communication (`{report.communication.score}/5`)")
            st.caption(f"Rating: **{report.communication.rating_label}**")
            st.progress(report.communication.score / 5.0)
            st.info(report.communication.feedback)

        with r_col2:
            st.markdown(f"#### 🎯 Correctness (`{report.correctness.score}/5`)")
            st.caption(f"Rating: **{report.correctness.rating_label}**")
            st.progress(report.correctness.score / 5.0)
            st.info(report.correctness.feedback)

        with r_col3:
            st.markdown(f"#### ⚡ Optimization (`{report.optimization.score}/5`)")
            st.caption(f"Rating: **{report.optimization.rating_label}**")
            st.progress(report.optimization.score / 5.0)
            st.info(report.optimization.feedback)

        st.markdown("---")
        feed_col1, feed_col2 = st.columns(2)
        with feed_col1:
            st.markdown("#### ✅ Demonstrated Strengths")
            for s in report.strengths:
                st.markdown(f"- {s}")

        with feed_col2:
            st.markdown("#### 🎯 Areas for Improvement")
            for imp in report.areas_for_improvement:
                st.markdown(f"- {imp}")

        with st.expander("🌟 View Ideal 5/5 Model Response Blueprint"):
            st.markdown(report.ideal_response_outline)

    # History Section
    if st.session_state.evaluation_history:
        with st.expander("📜 Session Evaluation History", expanded=False):
            st.table(st.session_state.evaluation_history)
