"""
LangGraph YAML Validation Tool
Router-Worker architecture with parallel validation nodes and synthesizer.
"""

import os
import yaml
import operator
from typing import TypedDict, Annotated, Any
from pydantic import BaseModel, Field
from pykwalify.core import Core
from pykwalify.errors import SchemaError
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------

class RuleResult(BaseModel):
    """Result of a single rule evaluation."""
    rule: str = Field(description="Short identifier for the rule checked")
    status: str = Field(description="PASS or FAIL")
    reason: str = Field(description="One-line explanation")


class WorkerValidationResponse(BaseModel):
    """Structured response from a validation worker."""
    results: list[RuleResult] = Field(description="List of rule evaluation results")


class SynthCheckItem(BaseModel):
    """A single check item in the synthesized review."""
    node: str = Field(description="Name of the validation node")
    status: str = Field(description="PASS or FAIL")
    summary: str = Field(description="One-line summary of what was checked")
    fix: str | None = Field(default=None, description="Actionable fix suggestion if FAIL, else null")


class SynthesizerResponse(BaseModel):
    """Structured synthesized review comment."""
    checks: list[SynthCheckItem] = Field(description="List of all validation check results")
    overall_verdict: str = Field(description="PASS if all checks passed, FAIL otherwise")
    developer_comment: str = Field(description="Concise developer-facing review comment with actionable suggestions")

# ---------------------------------------------------------------------------
# State definitions
# ---------------------------------------------------------------------------

class WorkerInput(TypedDict):
    """Input sent to each parallel validation worker."""
    worker_name: str
    yaml_slice: dict
    rules: dict


class ValidationState(TypedDict):
    """Main graph state."""
    yaml_data: dict
    schema_path: str
    rules_dir: str
    schema_valid: bool
    schema_errors: list[str]
    # Reducer: parallel workers append results into this list
    worker_results: Annotated[list[dict], operator.add]
    synthesized_comment: str


# ---------------------------------------------------------------------------
# LLM — lazy-initialized singleton (token-efficient model config)
# ---------------------------------------------------------------------------

_llm = None

def _get_llm() -> ChatOpenAI:
    """Lazy init so module can be imported before .env is loaded."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=512,       # cap response length for cost control
        )
    return _llm

# Compact system prompt shared by all workers
WORKER_SYSTEM_PROMPT = (
    "You are a YAML config validator. Given RULES and a YAML SLICE, "
    "evaluate each rule and return structured results."
)


# ---------------------------------------------------------------------------
# Helper: call LLM for a worker (structured output)
# ---------------------------------------------------------------------------

def _run_llm_validation(worker_name: str, rules: dict, yaml_slice: dict) -> dict:
    """Send a token-efficient prompt to GPT-4o with structured output."""
    structured_llm = _get_llm().with_structured_output(WorkerValidationResponse)
    human_msg = (
        f"RULES:\n{yaml.dump(rules, default_flow_style=True, width=1000)}\n"
        f"YAML_SLICE:\n{yaml.dump(yaml_slice, default_flow_style=True, width=1000)}"
    )
    response: WorkerValidationResponse = structured_llm.invoke([
        SystemMessage(content=WORKER_SYSTEM_PROMPT),
        HumanMessage(content=human_msg),
    ])
    return {
        "worker": worker_name,
        "response": response,
    }


# ---------------------------------------------------------------------------
# Node: Schema Validation (Gate)
# ---------------------------------------------------------------------------

def schema_validation(state: ValidationState) -> dict:
    """Validate YAML against pykwalify schema. Acts as a gate."""
    try:
        c = Core(
            source_data=state["yaml_data"],
            schema_files=[state["schema_path"]],
        )
        c.validate(raise_exception=True)
        return {"schema_valid": True, "schema_errors": []}
    except SchemaError as e:
        errors = [str(err) for err in e.errors] if hasattr(e, "errors") else [str(e)]
        return {"schema_valid": False, "schema_errors": errors}


# ---------------------------------------------------------------------------
# Conditional edge: proceed or fail early
# ---------------------------------------------------------------------------

def should_continue(state: ValidationState) -> list[Send] | str:
    """If schema is invalid, go to END. Otherwise fan-out to workers."""
    if not state.get("schema_valid", False):
        return "synthesizer"  # go straight to synthesizer with failure info

    yaml_data = state["yaml_data"]
    rules_dir = state["rules_dir"]
    identifier = yaml_data.get("dataset", {}).get("identifier", {})
    source = yaml_data.get("dataset", {}).get("source", {})
    target = yaml_data.get("dataset", {}).get("target", {})

    # ---- Build minimal YAML slices per worker ----

    # SchemaChange worker needs: targetmode, overwriteSchema, src_overwriteSchema, schemaChange flag
    sc_slice = {
        "targetmode": identifier.get("targetmode"),
        "overwriteSchema": identifier.get("overwriteSceham"),  # matches schema.yml key
        "src_overwriteSchema": identifier.get("src_overwriteScehma"),
        "schemaChange": identifier.get("schemaChagne"),
    }

    # LoadType worker needs: loadtype, checkpoint/schema locations
    lt_slice = {
        "loadtype": identifier.get("loadtype"),
        "ODSCheckpointLocation": identifier.get("ODSchekcpointLocation"),
        "ODSschemaLocation": identifier.get("ODSschemaLocation"),
    }

    # FileType worker needs: fileformat + source options
    ft_slice = {
        "fileformat": identifier.get("datainjection", {}).get("fileformat"),
        "source_options": source.get("options", []),
        "src_schema_enforcement": identifier.get("datainjection", {}).get("src_schema_enforcement"),
    }

    # Tags worker needs: targetmode + target columns
    tags_slice = {
        "targetmode": identifier.get("targetmode"),
        "target_columns": target.get("columns", []),
    }

    workers = [
        Send("schema_change_worker", WorkerInput(
            worker_name="SchemaChange",
            yaml_slice=sc_slice,
            rules=_load_rule_file(rules_dir, "schema_change_rules.yml"),
        )),
        Send("load_type_worker", WorkerInput(
            worker_name="LoadType",
            yaml_slice=lt_slice,
            rules=_load_rule_file(rules_dir, "load_type_rules.yml"),
        )),
        Send("file_type_worker", WorkerInput(
            worker_name="FileType",
            yaml_slice=ft_slice,
            rules=_load_rule_file(rules_dir, "file_type_rules.yml"),
        )),
        Send("tags_worker", WorkerInput(
            worker_name="Tags",
            yaml_slice=tags_slice,
            rules=_load_rule_file(rules_dir, "tags_rules.yml"),
        )),
    ]
    return workers


def _load_rule_file(rules_dir: str, filename: str) -> dict:
    """Load a single rule YAML file."""
    path = os.path.join(rules_dir, filename)
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Worker nodes (parallel)
# ---------------------------------------------------------------------------

def schema_change_worker(state: WorkerInput) -> dict:
    result = _run_llm_validation(state["worker_name"], state["rules"], state["yaml_slice"])
    return {"worker_results": [result]}


def load_type_worker(state: WorkerInput) -> dict:
    result = _run_llm_validation(state["worker_name"], state["rules"], state["yaml_slice"])
    return {"worker_results": [result]}


def file_type_worker(state: WorkerInput) -> dict:
    result = _run_llm_validation(state["worker_name"], state["rules"], state["yaml_slice"])
    return {"worker_results": [result]}


def tags_worker(state: WorkerInput) -> dict:
    result = _run_llm_validation(state["worker_name"], state["rules"], state["yaml_slice"])
    return {"worker_results": [result]}


# ---------------------------------------------------------------------------
# Synthesizer node
# ---------------------------------------------------------------------------

SYNTH_SYSTEM_PROMPT = (
    "You are a code reviewer. Given validation results from multiple checks on a "
    "YAML config file, produce a concise developer-facing review with actionable fixes."
)


def synthesizer(state: ValidationState) -> dict:
    """Aggregate all results and produce a structured developer comment."""
    # If schema failed, produce an early-exit comment
    if not state.get("schema_valid", False):
        errors = state.get("schema_errors", [])
        comment = "❌ **Schema Validation Failed** — fix these before other checks can run:\n"
        for err in errors:
            comment += f"  - {err}\n"
        return {"synthesized_comment": comment}

    # Build compact summary from structured worker results
    worker_data = state.get("worker_results", [])
    lines = []
    for wr in worker_data:
        resp: WorkerValidationResponse = wr["response"]
        for r in resp.results:
            lines.append(f"[{wr['worker']}] {r.rule}: {r.status} — {r.reason}")
    summary = "\n".join(lines)

    structured_llm = _get_llm().with_structured_output(SynthesizerResponse)
    response: SynthesizerResponse = structured_llm.invoke([
        SystemMessage(content=SYNTH_SYSTEM_PROMPT),
        HumanMessage(content=f"VALIDATION RESULTS:\n{summary}"),
    ])
    return {"synthesized_comment": _format_synth_response(response)}


def _format_synth_response(resp: SynthesizerResponse) -> str:
    """Convert structured synthesizer response to a readable developer comment."""
    lines = []
    for check in resp.checks:
        icon = "✅" if check.status == "PASS" else "❌"
        lines.append(f"{icon} **{check.node}** — {check.summary}")
        if check.fix:
            lines.append(f"   💡 Fix: {check.fix}")
    lines.append("")
    verdict_icon = "✅" if resp.overall_verdict == "PASS" else "❌"
    lines.append(f"**Overall: {verdict_icon} {resp.overall_verdict}**")
    lines.append("")
    lines.append(resp.developer_comment)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_validation_graph() -> StateGraph:
    """Construct the LangGraph validation workflow."""
    graph = StateGraph(ValidationState)

    # Nodes
    graph.add_node("schema_validation", schema_validation)
    graph.add_node("schema_change_worker", schema_change_worker)
    graph.add_node("load_type_worker", load_type_worker)
    graph.add_node("file_type_worker", file_type_worker)
    graph.add_node("tags_worker", tags_worker)
    graph.add_node("synthesizer", synthesizer)

    # Edges
    graph.add_edge(START, "schema_validation")
    graph.add_conditional_edges("schema_validation", should_continue,
                                ["schema_change_worker", "load_type_worker",
                                 "file_type_worker", "tags_worker", "synthesizer"])
    graph.add_edge("schema_change_worker", "synthesizer")
    graph.add_edge("load_type_worker", "synthesizer")
    graph.add_edge("file_type_worker", "synthesizer")
    graph.add_edge("tags_worker", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()
