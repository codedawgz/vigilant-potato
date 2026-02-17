from __future__ import annotations

import argparse
import json
from typing import Annotated, Any, Dict, Literal, Optional, TypedDict

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from pykwalify.core import Core


def _merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class AgentState(TypedDict, total=False):
    yaml_data: Dict[str, Any]
    schema_data: Dict[str, Any]
    rules_data: Dict[str, Any]
    schema_change_flag: bool
    schema_change_message: str
    Schema_change_flag: bool
    schema_Change_message: str
    schema_valid: bool
    schema_error: Optional[str]
    worker_results: Annotated[Dict[str, Dict[str, Any]], _merge_dicts]
    final_comment: str


class RuleIssue(BaseModel):
    path: str = Field(description="YAML path of the issue.")
    issue: str = Field(description="What is wrong.")
    fix: str = Field(description="How to fix it.")


class RuleValidationResponse(BaseModel):
    status: Literal["pass", "fail"]
    summary: str
    issues: list[RuleIssue] = Field(default_factory=list)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        parsed = yaml.safe_load(f) or {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected YAML object in {path}, got {type(parsed).__name__}")
    return parsed


def _schema_failure_comment(error_message: str) -> str:
    return (
        "Schema validation failed, so rule validation was skipped.\n"
        "Please fix the YAML structure first.\n\n"
        f"PyKwalify error:\n{error_message}"
    )


def _build_worker_context(
    worker_name: str,
    yaml_data: Dict[str, Any],
    schema_change_message: str,
) -> Dict[str, Any]:
    dataset = yaml_data.get("dataset", {}) if isinstance(yaml_data, dict) else {}
    identifier = dataset.get("identifier", {}) if isinstance(dataset, dict) else {}
    source = dataset.get("source", {}) if isinstance(dataset, dict) else {}
    target = dataset.get("target", {}) if isinstance(dataset, dict) else {}
    file_type = (
        identifier.get("datainjection", {}).get("fileformat")
        if isinstance(identifier.get("datainjection"), dict)
        else None
    )

    contexts = {
        "schemaChange": {
            "schema_change_message": schema_change_message,
            "target_mode": identifier.get("targetmode"),
            "source": source,
            "target": target,
            "identifier": identifier,
        },
        "LoadType": {
            "load_type": identifier.get("loadtype"),
            "identifier": identifier,
        },
        "FileType": {
            "file_type": file_type,
            "source_options": source.get("options") if isinstance(source, dict) else None,
            "identifier": identifier,
            "source": source,
        },
        "Tags": {
            "target_mode": identifier.get("targetmode"),
            "target_columns": target.get("columns") if isinstance(target, dict) else None,
            "target": target,
            "identifier": identifier,
        },
    }
    return contexts.get(worker_name, {"yaml_data": yaml_data})


def _validate_rule_with_llm(
    llm: ChatOpenAI,
    worker_name: str,
    worker_rules: Dict[str, Any],
    yaml_data: Dict[str, Any],
    schema_change_message: str,
) -> Dict[str, Any]:
    validator = llm.with_structured_output(RuleValidationResponse)
    worker_context = _build_worker_context(worker_name, yaml_data, schema_change_message)
    messages = [
        SystemMessage(
            content=(
                "You validate YAML against a provided rule subtree. "
                "Return only data requested by the output schema. "
                "If any rule is violated, set status=fail and include specific actionable issues."
            )
        ),
        HumanMessage(
            content=(
                f"Rule root: {worker_name}\n\n"
                f"Rules subtree:\n{json.dumps(worker_rules, indent=2)}\n\n"
                f"Relevant YAML/context:\n{json.dumps(worker_context, indent=2)}\n\n"
                "Validate strictly against the rules subtree."
            )
        ),
    ]
    response = validator.invoke(messages)
    return response.model_dump()


def _schema_validation_node(state: AgentState) -> AgentState:
    try:
        core = Core(source_data=state["yaml_data"], schema_data=state["schema_data"])
        core.validate(raise_exception=True)
    except Exception as exc:  # pykwalify raises several exception types
        message = str(exc)
        return {
            "schema_valid": False,
            "schema_error": message,
            "final_comment": _schema_failure_comment(message),
        }
    return {"schema_valid": True, "schema_error": None}


def _route_after_schema(state: AgentState) -> str:
    return "router" if state.get("schema_valid") else "end"


def _router_node(state: AgentState) -> AgentState:
    return {}


def _route_rule_workers(state: AgentState) -> list[str]:
    routes = ["validate_load_type", "validate_file_type", "validate_tags"]
    schema_change_flag = state.get("schema_change_flag")
    if schema_change_flag is None:
        schema_change_flag = state.get("Schema_change_flag", False)
    if schema_change_flag:
        routes.append("validate_schema_change")
    return routes


def _rule_worker(worker_name: str, llm: ChatOpenAI):
    def _node(state: AgentState) -> AgentState:
        rules = state.get("rules_data", {})
        worker_rules = rules.get(worker_name)
        if worker_rules is None:
            result = {
                "status": "pass",
                "summary": f"No rules found for '{worker_name}', skipped.",
                "issues": [],
            }
            return {"worker_results": {worker_name: result}}

        try:
            result = _validate_rule_with_llm(
                llm=llm,
                worker_name=worker_name,
                worker_rules=worker_rules,
                yaml_data=state.get("yaml_data", {}),
                schema_change_message=(
                    state.get("schema_change_message")
                    or state.get("schema_Change_message", "")
                ),
            )
        except Exception as exc:
            result = {
                "status": "fail",
                "summary": f"{worker_name} validation failed due to runtime error.",
                "issues": [
                    {
                        "path": "$",
                        "issue": f"LLM validation error: {str(exc)}",
                        "fix": "Check OpenAI credentials/model availability and retry.",
                    }
                ],
            }

        return {"worker_results": {worker_name: result}}

    return _node


def _synthesizer_node(llm: ChatOpenAI):
    def _node(state: AgentState) -> AgentState:
        worker_results = state.get("worker_results", {})
        synth_messages = [
            SystemMessage(
                content=(
                    "You create a layman-readable validation summary for a YAML file. "
                    "Focus on concrete issues and exact fixes. Be concise."
                )
            ),
            HumanMessage(
                content=(
                    "Validation results by rule worker:\n"
                    f"{json.dumps(worker_results, indent=2)}\n\n"
                    "Write a final comment with:\n"
                    "1) overall status\n"
                    "2) numbered list of issues\n"
                    "3) what to fix in YAML paths.\n"
                    "If no issues, clearly say all validations passed."
                )
            ),
        ]
        try:
            response = llm.invoke(synth_messages)
            final_comment = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
        except Exception as exc:
            failed_workers = [
                name
                for name, result in worker_results.items()
                if result.get("status") == "fail"
            ]
            if failed_workers:
                final_comment = (
                    "Validation completed with failures.\n"
                    f"Failed checks: {', '.join(failed_workers)}\n"
                    f"Synthesis error: {str(exc)}"
                )
            else:
                final_comment = (
                    "All validations passed, but synthesis failed.\n"
                    f"Synthesis error: {str(exc)}"
                )

        return {"final_comment": final_comment}

    return _node


def build_validation_graph(model: str = "gpt-4.1-mini"):
    llm = ChatOpenAI(model=model, temperature=0)
    graph = StateGraph(AgentState)

    graph.add_node("schema_validation", _schema_validation_node)
    graph.add_node("router", _router_node)
    graph.add_node("validate_schema_change", _rule_worker("schemaChange", llm))
    graph.add_node("validate_load_type", _rule_worker("LoadType", llm))
    graph.add_node("validate_file_type", _rule_worker("FileType", llm))
    graph.add_node("validate_tags", _rule_worker("Tags", llm))
    graph.add_node("synthesizer", _synthesizer_node(llm))

    graph.add_edge(START, "schema_validation")
    graph.add_conditional_edges(
        "schema_validation",
        _route_after_schema,
        {"router": "router", "end": END},
    )
    graph.add_conditional_edges("router", _route_rule_workers)

    graph.add_edge("validate_schema_change", "synthesizer")
    graph.add_edge("validate_load_type", "synthesizer")
    graph.add_edge("validate_file_type", "synthesizer")
    graph.add_edge("validate_tags", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


def run_validation_agent(
    input_yaml_path: str,
    schema_yaml_path: str,
    rules_yaml_path: str,
    schema_change_flag: bool,
    schema_change_message: str,
    model: str = "gpt-4.1-mini",
) -> AgentState:
    app = build_validation_graph(model=model)
    final_state = app.invoke(
        {
            "yaml_data": _load_yaml(input_yaml_path),
            "schema_data": _load_yaml(schema_yaml_path),
            "rules_data": _load_yaml(rules_yaml_path),
            "schema_change_flag": schema_change_flag,
            "schema_change_message": schema_change_message,
            "Schema_change_flag": schema_change_flag,
            "schema_Change_message": schema_change_message,
            "worker_results": {},
        }
    )
    return final_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SD YAML with LangGraph + OpenAI + PyKwalify.")
    parser.add_argument("--input-yaml", default="sd.yml")
    parser.add_argument("--schema-yaml", default="schema.yml")
    parser.add_argument("--rules-yaml", default="rules.yaml")
    parser.add_argument(
        "--schema-change-flag",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set true when schema change is detected.",
    )
    parser.add_argument(
        "--schema-change-message",
        default="",
        help="Detailed schema change report used by schemaChange rule worker.",
    )
    parser.add_argument("--model", default="gpt-4.1-mini")
    args = parser.parse_args()

    result = run_validation_agent(
        input_yaml_path=args.input_yaml,
        schema_yaml_path=args.schema_yaml,
        rules_yaml_path=args.rules_yaml,
        schema_change_flag=args.schema_change_flag,
        schema_change_message=args.schema_change_message,
        model=args.model,
    )

    print("=== Final Comment ===")
    print(result.get("final_comment", "No comment generated."))
    print("\n=== Worker Results ===")
    print(json.dumps(result.get("worker_results", {}), indent=2))


if __name__ == "__main__":
    main()
