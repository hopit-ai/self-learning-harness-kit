"""Evidence-first behavioural contrasts built only from public STREAM episodes.

DISCLOSURE: _PREREQUISITES is a human-written policy scaffold, not a learned
cause, benchmark label, or mechanically verified prerequisite. Missing-action
operation inference from requests is also heuristic. Consent/error-text flags
are hypotheses only and do not partition clusters. Bundles disclose these
choices and any compacted or omitted public evidence.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slh.learner.aliases import ALIAS, AliasTable
from slh.learner.triage import (
    COMPONENTS,
    Evidence,
    heuristic_signatures,
    is_write,
    observed_signatures,
    public_path,
)
from slh.trace import check_no_grading_fields

REPO = Path(__file__).resolve().parents[2]
POLICY = REPO / "vendor/tau2-bench/data/tau2/domains/retail/policy.md"
TOOLS = REPO / "vendor/tau2-bench/src/tau2/domains/retail/tools.py"
_WRITE_OPERATIONS = {
    "return_delivered_order_items": "return",
    "exchange_delivered_order_items": "exchange",
    "modify_pending_order_items": "modify",
    "modify_pending_order_address": "modify",
    "modify_pending_order_payment": "modify",
    "cancel_pending_order": "cancel",
    "modify_user_address": "update",
    "transfer_to_human_agents": "transfer",
}
MAX_EXAMPLE_BYTES = 10_000
# Amendment 3 (pre-registration adjustment to the observed sizes of complete
# dialogues): the per-example limit of tiered packets. Amendment-1/2 packets
# keep MAX_EXAMPLE_BYTES.
TIERED_MAX_EXAMPLE_BYTES = 24_000
MAX_ASSISTANT_CHARS = 4_000
MAX_THOUGHTS = 2
MAX_THOUGHT_CHARS = 600
_PREREQUISITES = {
    "cancel": "pending_status_and_action_consent",
    "modify": "pending_status_selection_payment_and_action_consent",
    "exchange": "delivered_status_item_mapping_payment_and_action_consent",
    "return": "delivered_status_refund_destination_and_action_consent",
    "update": "authenticated_owner_and_action_consent",
    "transfer": "policy_scope",
    "information": "authenticated_owner",
}


def call_operation(name: str) -> str | None:
    return _WRITE_OPERATIONS.get(name)


def operations(episode: Evidence) -> tuple[str, ...]:
    called = {call_operation(c["name"]) for m in episode.messages for c in m["tool_calls"]}
    called.discard(None)
    if called:
        return tuple(sorted(called))
    # Only for missing-action episodes: identify the needed public write tool.
    request = " ".join(m["content"] for m in episode.messages if m["role"] == "user")
    needed = {op for tool, op in _WRITE_OPERATIONS.items() if tool in request}
    patterns = {"cancel": r"\bcancel\w*\b", "exchange": r"\bexchange\w*\b",
                "return": r"\b(?:return\w*|refund\w*)\b",
                "modify": r"\b(?:modify|change)\b.*\border\b",
                "update": r"\b(?:update|change|modify)\b.*\b(?:default|profile|my) address\b",
                "transfer": r"\b(?:transfer|human agent)\b"}
    needed.update(op for op, pattern in patterns.items() if re.search(pattern, request, re.IGNORECASE))
    return tuple(sorted(needed or {"information"}))


@dataclass
class BehaviouralCluster:
    operation: str
    prerequisite: str
    mechanism: str  # Observable signal, never a gold-action category.
    target_signature: str  # Compatibility with the loop's local lesson provenance.
    episodes: list[Evidence] = field(default_factory=list)
    mixed_tasks: int = 0
    passing_tasks: set[str] = field(default_factory=set)

    @property
    def id(self) -> str:
        return f"{self.operation}/{self.prerequisite}/{self.mechanism}"

    @property
    def failures(self) -> list[Evidence]:
        return [e for e in self.episodes if e.reward != 1]


def behavioural_clusters(episodes: list[Evidence]) -> list[BehaviouralCluster]:
    """Families match comparisons; they never define a behavioural cluster."""
    if any(e.split_name != "stream" for e in episodes):
        raise ValueError("only stream episodes may become writer evidence")
    split = json.loads(public_path(REPO / "splits/retail.json").read_text(encoding="utf-8"))
    held_out = set(split["dev"] + split["test"] + split["confirm"])
    if any(e.task_id in held_out for e in episodes):
        raise ValueError("held-out task IDs cannot be relabelled as stream evidence")
    keys = [(e.run_manifest, e.episode_id) for e in episodes]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate source episodes cannot supply distinct contrasts")
    task_outcomes: dict[str, set[bool]] = defaultdict(set)
    for episode in episodes:
        task_outcomes[episode.task_id].add(episode.reward == 1)
    clusters: dict[tuple[str, str, str], BehaviouralCluster] = {}
    for episode in episodes:
        if episode.reward == 1:
            continue
        signals = tuple(s for s in observed_signatures(episode)
                        if s not in {"tool_error", "write_without_explicit_confirmation", "unclassified_failure"})
        # Text heuristics stay context/uncertainty; they never label a cause.
        mechanisms = signals or ("component_failure",)
        for operation in operations(episode):
            prerequisite = _PREREQUISITES.get(operation, "policy_prerequisites")
            for mechanism in mechanisms:
                key = (operation, prerequisite, mechanism)
                cluster = clusters.setdefault(key, BehaviouralCluster(*key, mechanism))
                cluster.episodes.append(episode)
    for cluster in clusters.values():
        cluster.passing_tasks = {t for t, outcomes in task_outcomes.items() if True in outcomes}
        cluster.episodes.extend(e for e in episodes if e.reward == 1 and
                                cluster.operation in operations(e))
        cluster.mixed_tasks = len({e.task_id for e in cluster.failures
                                   if len(task_outcomes[e.task_id]) == 2})
    return sorted(clusters.values(), key=lambda c: (-c.mixed_tasks, -len(c.failures), c.id))


def decisive_slice(episode: Evidence, operation: str) -> list[dict[str, Any]]:
    """Select semantic trajectory steps, including every relevant call and result.

    User turns include refusals/conditional replies verbatim. 'consent_reply'
    names a location in the sequence, not a claim that it authorized an action.
    """
    messages = episode.messages
    first_user = next((i for i, m in enumerate(messages) if m["role"] == "user"), 0)
    selected: dict[int, str] = {first_user: "request"} if messages and messages[first_user]["role"] == "user" else {}
    actions = []
    lookups = []

    def preceding_user(index: int, stage: str) -> int | None:
        user = next((j for j in range(index - 1, first_user - 1, -1)
                     if messages[j]["role"] == "user"), None)
        if user is not None:
            selected[user] = "request" if user == first_user else stage
            if user > first_user and messages[user - 1]["role"] == "assistant" and messages[user - 1]["content"]:
                selected[user - 1] = "proposal_or_question"
        return user

    for i, message in enumerate(messages):
        if i < first_user:
            continue
        calls = [c for c in message["tool_calls"] if c["name"] != "think"]
        relevant = calls  # Every write can affect the DB outcome, even across operations.
        if relevant:
            selected[i] = "action" if any(is_write(c["name"]) or
                                          c["name"] == "transfer_to_human_agents"
                                          for c in relevant) else "lookup"
            (actions if selected[i] == "action" else lookups).append(i)
            user = preceding_user(i, "consent_reply" if selected[i] == "action" else "lookup_input")
            if selected[i] == "action" and user is not None:
                # Retain the user refinement that prompted the final proposal too.
                proposal = next((j for j in range(user - 1, first_user - 1, -1)
                                 if messages[j]["role"] == "assistant" and messages[j]["content"]), None)
                if proposal is not None:
                    selected[proposal] = "proposal_or_question"
                    preceding_user(proposal, "request_refinement")
            # Results belong to the call, including an error or a negative result.
            j = i + 1
            while j < len(messages) and messages[j]["role"] == "tool":
                selected[j] = "result"
                j += 1
        if message["role"] == "tool" and (i == 0 or i - 1 in selected):
            selected[i] = "result"
    # Missing-action/communication failures: show the first post-lookup
    # proposal/reply and the terminal unresolved reply, not every loop iteration.
    if not actions:
        after = lookups[-1] if lookups else first_user
        proposal = next((j for j in range(after + 1, len(messages))
                         if messages[j]["role"] == "assistant" and messages[j]["content"]), None)
        if proposal is not None:
            selected[proposal] = "proposal_or_question"
            reply = next((j for j in range(proposal + 1, len(messages))
                          if messages[j]["role"] == "user"), None)
            if reply is not None:
                selected[reply] = "user_reply"
        preceding_user(len(messages), "unresolved_reply")
    if messages and messages[-1]["role"] == "assistant" and messages[-1]["content"]:
        selected[len(messages) - 1] = "completion"
    # Repeated identical read/result pairs carry no new lookup facts. Keep
    # the first and last occurrence so the repeated-call signal stays visible.
    reads: dict[str, list[int]] = defaultdict(list)
    for index in lookups:
        results = []
        j = index + 1
        while j < len(messages) and messages[j]["role"] == "tool":
            results.append(messages[j]["content"])
            j += 1
        key = json.dumps({"content": messages[index]["content"], "calls": [{"name": c["name"], "arguments": c["arguments"]}
                                    for c in messages[index]["tool_calls"]],
                          "results": results}, sort_keys=True)
        reads[key].append(index)
    for indices in reads.values():
        for index in indices[1:-1]:
            # Call results can repeat while accompanying questions carry new
            # meaning. Never discard substantive assistant dialogue.
            if messages[index]["content"]:
                continue
            selected.pop(index, None)
            j = index + 1
            while j < len(messages) and messages[j]["role"] == "tool":
                selected.pop(j, None)
                j += 1
    result = []
    for index, stage in sorted(selected.items()):
        message = messages[index]
        result.append({"message_index": index, "stage": stage, "role": message["role"],
                       "content": (message["content"][:MAX_ASSISTANT_CHARS]
                                   if message["role"] == "assistant" else message["content"]),
                       **({"content_truncated": True, "original_content_chars": len(message["content"])}
                          if message["role"] == "assistant" and
                          len(message["content"]) > MAX_ASSISTANT_CHARS else {}), "tool_calls": [
                           {"name": c["name"], "arguments": c["arguments"]}
                           for c in message["tool_calls"] if c["name"] != "think"]})
    return result


def compact_lookups(sliced: list[dict[str, Any]], result_tools: dict[int, str] | None = None) -> None:
    """Retain action-relevant catalogue variants, plus a bounded sample digest.

    This projection is explicitly lossy; raw lookup calls, action arguments,
    requests, consent and write-result action facts remain intact. No DB is opened.
    """
    context = " ".join(m["content"] for m in sliced if m["role"] != "tool")
    arguments = json.dumps([c["arguments"] for m in sliced for c in m["tool_calls"]])
    latest_tool = None
    order_snapshots: dict[str, tuple[int, dict[str, Any]]] = {}
    action_operations = {call_operation(c["name"]) for m in sliced for c in m["tool_calls"]}
    modifying_address = any(c["name"] in {"modify_user_address", "modify_pending_order_address"}
                            for m in sliced for c in m["tool_calls"])
    for message in sliced:
        if message["tool_calls"]:
            latest_tool = message["tool_calls"][0]["name"]
        if message["role"] != "tool":
            continue
        try:
            result = json.loads(message["content"])
        except ValueError:
            continue
        if not isinstance(result, dict) or result.get("error"):
            continue
        if "order_id" in result and "status" in result:
            # Full order snapshots include shipping details and many null fields
            # unrelated to the action. Keep every non-null operation field, item
            # mapping, payment history, ownership and status, including writes.
            previous = order_snapshots.get(str(result["order_id"]))
            prior = previous[1] if previous else {}
            omitted = [key for key, value in result.items() if (
                value is None and key not in prior and
                not any(key.startswith(op + "_") for op in action_operations if op)) or
                key == "fulfillments" or key == "address" and not modifying_address]
            snapshot = {k: v for k, v in result.items() if k not in omitted}
            if previous:
                prior_index, prior = previous
                unchanged = [k for k, v in snapshot.items() if k not in {"order_id", "status"}
                             and k in prior and prior[k] == v]
                message["baseline_message_index"] = prior_index
                removed = sorted(prior.keys() - snapshot.keys())
                if removed:
                    message["removed_result_fields"] = removed
                message["content"] = json.dumps({k: v for k, v in snapshot.items() if k not in unchanged})
                message["projection"] = "order_delta"
            else:
                message["content"] = json.dumps(snapshot)
                message["projection"] = "order_fields"
            order_snapshots[str(result["order_id"])] = (message["message_index"], snapshot)
            continue
        tool = (result_tools or {}).get(message["message_index"], latest_tool)
        if tool != "get_product_details" or not isinstance(result.get("variants"), dict):
            continue
        variants = result["variants"]
        concrete = {key: value for key, value in variants.items()
                    if str(key) in context or str(key) in arguments}
        mentioned: dict[str, set[str]] = defaultdict(set)
        for value in variants.values():
            for option, choice in value.get("options", {}).items():
                if re.search(r"(?<!\w)" + re.escape(str(choice)) + r"(?!\w)", context, re.IGNORECASE):
                    mentioned[option].add(str(choice))
        discussed = {key: value for key, value in variants.items() if mentioned and
                     all(str(value.get("options", {}).get(option)) in choices
                         for option, choices in mentioned.items()) and key not in concrete}
        relevant = {**concrete, **dict(list(discussed.items())[:6])}
        sample = [value for key, value in variants.items() if key not in relevant][:3]
        message["content"] = json.dumps({"name": result.get("name"),
            "product_id": result.get("product_id"), "variants": relevant,
            "variant_digest": {"total": len(variants), "sample": sample,
                               "discussed_matches": len(discussed),
                               "omitted": len(variants) - len(relevant) - len(sample)}})
        message["projection"] = "catalogue_variants"


def policy_excerpt(operation: str, path: Path = POLICY) -> dict[str, Any]:
    text = public_path(path).read_text(encoding="utf-8")
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    general = [s for s in sections if s.startswith(("# Retail", "## Generic", "## Domain"))]
    specific = [s for s in sections if re.match(r"## " + re.escape(operation), s, re.IGNORECASE)]
    if operation == "information":
        specific = []
    return {"source": "tau2 retail public policy.md", "sha256": hashlib.sha256(
        text.encode()).hexdigest(), "text": "\n".join(general + specific)}


def tool_contracts(names: set[str], path: Path = TOOLS) -> list[dict[str, Any]]:
    """Extract public signatures/docstrings without executing tools or loading a DB."""
    source = public_path(path).read_text(encoding="utf-8")
    result = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef) or node.name not in names:
            continue
        if not any(isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and
                   d.func.id == "is_tool" for d in node.decorator_list):
            continue
        result.append({"tool": node.name, "parameters": {a.arg: ast.unparse(a.annotation)
                       if a.annotation else "unknown" for a in node.args.args if a.arg != "self"},
                       "returns": ast.unparse(node.returns) if node.returns else "unknown",
                       "contract": ast.get_docstring(node) or "",
                       "source": "tau2 retail public tools.py", "sha256": hashlib.sha256(
                           source.encode()).hexdigest()})
    return result


def select_contrasts(cluster: BehaviouralCluster, limit: int = 2) -> tuple[list[tuple], list[Evidence], list[Evidence]]:
    if not 1 <= limit <= 2:
        raise ValueError("select at most two contrast pairs")
    failures = cluster.failures
    successes = [e for e in cluster.episodes if e.reward == 1]
    success_tasks = {s.task_id for s in successes}
    # Prefer mixed tasks, then concise decisive traces within each priority.
    sizes = {id(e): len(json.dumps(decisive_slice(e, cluster.operation)))
             for e in cluster.episodes}
    failures = sorted(failures, key=lambda e: (e.task_id not in success_tasks, sizes[id(e)]))
    successes = sorted(successes, key=lambda e: sizes[id(e)])
    pairs: list[tuple[Evidence, Evidence, str]] = []
    used_failed: set[tuple[str, str]] = set()
    used_success: set[tuple[str, str]] = set()
    # Exhaust same-task matches before considering any same-family fallback.
    for match in ("same_task", "same_family"):
        for failed in failures:
            fk = (failed.run_manifest, failed.episode_id)
            if fk in used_failed:
                continue
            successful = next((s for s in successes if
                (s.run_manifest, s.episode_id) not in used_success and
                (s.task_id == failed.task_id if match == "same_task" else
                 failed.family is not None and failed.family == s.family)), None)
            if successful is None:
                continue
            pairs.append((failed, successful, match))
            used_failed.add(fk)
            used_success.add((successful.run_manifest, successful.episode_id))
            if len(pairs) == limit:
                break
        if len(pairs) == limit:
            break
    never_pass = [e for e in failures if e.task_id not in (cluster.passing_tasks | success_tasks)][:2]
    # Successful examples are negative controls, even when their own heuristic fires.
    controls = [s for s in successes if (s.run_manifest, s.episode_id) not in used_success][:2]
    if not controls:
        controls = successes[:1]  # A paired success also tests a proposed intervention.
    return pairs, never_pass, controls


def _project_example(episode: Evidence, example: dict[str, Any]) -> None:
    """Alias-project and compact one example's slice and thoughts in place."""
    aliases = AliasTable()
    structured_sources = [message["content"] for message in episode.messages
                          if message["role"] == "tool"]
    structured_sources.extend(call["arguments"] for message in episode.messages
                              for call in message["tool_calls"] if call["name"] != "think")
    aliases.prime(structured_sources)
    for message in episode.messages:
        aliases.collect_text(message["content"])
        for call in message["tool_calls"]:
            if call["name"] != "think":
                aliases.collect(call["arguments"])
    for thought in episode.public_think:
        aliases.collect_text(thought["content"])
    call_names = {c["id"]: c["name"] for m in episode.messages for c in m["tool_calls"] if "id" in c}
    result_tools = {i: call_names[m["tool_call_id"]] for i, m in enumerate(episode.messages)
                    if m.get("tool_call_id") in call_names}
    compact_lookups(example["slice"], result_tools)
    # Project evidence values only, never policy, roles or audit metadata.
    for message in example["slice"]:
        message["content"] = aliases.project(message["content"])
        if message["role"] == "tool":
            try:
                structured = json.loads(message["content"])
            except ValueError:
                structured = None
            if isinstance(structured, (dict, list)):
                message["content"] = structured
        for call in message["tool_calls"]:
            call["arguments"] = aliases.project(call["arguments"])
    for thought in example["public_think"]:
        text = aliases.project(thought["content"])
        thought["content"] = text[:MAX_THOUGHT_CHARS]
        thought["truncated"] = len(text) > MAX_THOUGHT_CHARS


def _legacy_bundle(cluster: BehaviouralCluster, notes: list[dict[str, Any]], *,
                 policy_path: Path = POLICY, tools_path: Path = TOOLS,
                 provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    pairs, never_pass, controls = select_contrasts(cluster)
    references: dict[tuple[str, str], str] = {}
    local: list[Evidence] = []

    def reference(episode: Evidence) -> str:
        key = (episode.run_manifest, episode.episode_id)
        if key not in references:
            references[key] = f"E{len(references) + 1}"
            local.append(episode)
        return references[key]

    contrasts = [{"match": match, "failed": reference(f), "successful": reference(s)}
                 for f, s, match in pairs]
    never = [reference(e) for e in never_pass]
    negative = [reference(e) for e in controls]
    if not local:
        for episode in cluster.failures[:2]:
            reference(episode)
    if provenance is not None:
        provenance.update({reference(e): {"run_manifest": e.run_manifest,
                                         "episode_id": e.episode_id, "task_id": e.task_id,
                                         "benchmark_family": e.family} for e in local})
    examples = []
    names = set()
    for episode in local:
        if any(t.get("sanitized") is not True for t in episode.public_think):
            raise ValueError("writer thoughts require an audited sanitized public projection")
        sliced = decisive_slice(episode, cluster.operation)
        names.update(c["name"] for m in sliced for c in m["tool_calls"])
        examples.append({"reference": reference(episode), "outcome": int(episode.reward == 1),
                         "numeric_reward": episode.reward,
                         "numeric_components": {k: v for k, v in episode.numeric_components.items()
                                                if k in COMPONENTS},
                         "components": {k: int(v == 1) for k, v in episode.components.items()
                                        if k in COMPONENTS},
                         "termination": episode.termination_reason, "slice": sliced,
                         "slice_coverage": {"total_messages": len(episode.messages),
                                            "selected_messages": len(sliced)},
                         "observed_signals": list(observed_signatures(episode, failures_only=False)),
                         "heuristics_not_labels": list(heuristic_signatures(episode)),
                         "heuristic_hypotheses": [{"signal": signal, "status": "hypothesis"}
                                                  for signal in heuristic_signatures(episode)],
                         "public_think_coverage": {"total": len(episode.public_think),
                                                   "selected": min(MAX_THOUGHTS, len(episode.public_think))},
                         "public_think": [{"turn": t.get("turn"), "index": t.get("index"),
                                           "content": t["content"], "source": {
                                               "file": "think_public.jsonl", "reference": reference(episode),
                                               "row": t.get("source_row")}, "sanitized": True}
                                          for t in episode.public_think[:MAX_THOUGHTS]]})
    applicable = [{"content": n["content"]} for n in notes if not n.get("target_signature") or
                  n["target_signature"] in {cluster.target_signature, cluster.id} or
                  n.get("scope") == "global"]
    payload = {"cluster_id": cluster.id, "operation": cluster.operation,
               "prerequisite": cluster.prerequisite, "observable_mechanism": cluster.mechanism,
               "target_signature": cluster.target_signature, "mixed_stream_tasks": cluster.mixed_tasks,
               "disclosures": {
                   "prerequisite_labels": "Human-written _PREREQUISITES; unverified policy scaffold, not causal labels.",
                   "operation_inference": "Called write tools; request heuristic only if none called.",
                   "heuristic_flags": "Unverified hypotheses; never cluster partitions.",
                   "lookup_projection": "Used/discussed variants + sample. Orders omit unrelated nulls, fulfillments and addresses (except address writes); ownership/items/payments/action fields kept. Deltas inherit absent fields from baseline_message_index; removed fields listed.",
                   "tool_result_encoding": "JSON results embedded as objects/lists; other results as text.",
                   "dialogue_projection": "Identical reads without dialogue: first/last kept. Assistant text including call commentary/proposals/questions retained up to the disclosed per-message cap; truncation flagged. Requests and every write kept.",
                   "example_byte_limit": MAX_EXAMPLE_BYTES,
                   "assistant_characters_each": MAX_ASSISTANT_CHARS,
                   "thought_limits": {"count": MAX_THOUGHTS, "characters_each": MAX_THOUGHT_CHARS}},
               "evidence_status": "sufficient" if len(pairs) == 2 else "insufficient_evidence",
               "evidence_reason": None if len(pairs) == 2 else "Two distinct same-task or same-family contrasts are required.",
               "alias_scope": "Per-example aliases; equal aliases across references do not imply equal identities",
               "evidence_hierarchy": [
                   {"kind": "public_policy_and_tool_contracts", "use": "normative support"},
                   {"kind": "visible_messages_calls_results", "use": "observable facts"},
                   {"kind": "numeric_stream_outcomes", "use": "outcomes, never causal labels"},
                   {"kind": "sanitized_public_think", "use": "agent hypothesis, not verified fact"},
                   {"kind": "heuristics", "use": "marked hypotheses, never training labels or cluster partitions"}],
               "pairs": contrasts, "never_pass_examples": never,
               "successful_negative_controls": negative, "examples": examples,
               "policy": policy_excerpt(cluster.operation, policy_path),
               "tool_contracts": tool_contracts(names, tools_path), "current_notes": applicable,
               "finding_chain": {
                   "observable_fact": [{"reference": e["reference"], "components": e["components"],
                                        "signals": e["observed_signals"]} for e in examples],
                   "suspected_mechanism": {"status": "hypothesis", "text": "Unresolved; infer from public evidence and policy only."},
                   "proposed_intervention": None,
                   "measured_effect": {"status": "not_measured", "value": None}}}
    check_no_grading_fields(payload)
    payload["missing_tool_contracts"] = sorted(names - {c["tool"] for c in payload["tool_contracts"]})
    if payload["tool_contracts"]:
        # All contracts come from one public source; retain its audit identity
        # once rather than repeat it for every function signature/docstring.
        payload["tool_contracts_source"] = {key: payload["tool_contracts"][0][key]
                                            for key in ("source", "sha256")}
        payload["tool_contracts"] = [{k: v for k, v in contract.items()
                                     if k not in {"source", "sha256"}}
                                    for contract in payload["tool_contracts"]]
    # Episodes have independent identity maps; collect the complete public trajectory
    # before projecting its slice so later lookups alias earlier requests as well.
    for episode, example in zip(local, examples, strict=True):
        _project_example(episode, example)
        example["projected_bytes"] = 0
        for _ in range(3):
            example["projected_bytes"] = len(json.dumps(example, separators=(",", ":")).encode("utf-8"))
        size = example["projected_bytes"]
        if size > MAX_EXAMPLE_BYTES:
            if payload["evidence_status"] == "sufficient":
                payload["evidence_status"] = "too_large"
                payload["evidence_reason"] = "Required example exceeds per-example byte budget."
            # Omit the whole oversized example; never deliver clipped arguments
            # or a partial action as apparently sufficient evidence.
            example["slice"] = []
            example["public_think"] = []
            example["budget_omission"] = True
    return payload


def build_bundle(cluster: BehaviouralCluster, notes: list[dict[str, Any]], *,
                 policy_path: Path = POLICY, tools_path: Path = TOOLS,
                 provenance: dict[str, Any] | None = None,
                 rules: EvidenceRules | None = None) -> dict[str, Any]:
    """The writer's evidence packet for one behavioural cluster.

    Without ``rules`` (or with the amendment-1 rules) this is the original
    two-contrast packet, byte for byte. With ``EvidenceRules.amendment3()`` it
    is the protocol amendment 3 packet: the strongest of four ordered evidence
    tiers, writer-facing labels and decisive-slice reduction of oversized
    examples (``tiered_bundle``)."""
    if rules is None or not rules.tiers:
        return _legacy_bundle(cluster, notes, policy_path=policy_path, tools_path=tools_path,
                              provenance=provenance)
    return tiered_bundle(cluster, notes, policy_path=policy_path, tools_path=tools_path,
                         provenance=provenance, reduce=rules.slice_reduction)


# --------------------------------------------------------------------------- amendment 3
#
# Evidence tiers (docs/protocol-amendment-3.md, "Evidence replacement"):
# matched_repeated > matched_single > cross_family_analogue >
# policy_grounded_singleton; otherwise insufficient_evidence. Every packet
# carries evidence_tier, references, match_criteria, missing_comparisons,
# confounders and mechanism_status: hypothesis.

TIERS = ("matched_repeated", "matched_single", "cross_family_analogue",
         "policy_grounded_singleton")
TIER_RANK = {tier: rank for rank, tier in enumerate(TIERS)}
TIER_LIMITATIONS = {
    "matched_repeated": "Distinct episodes do not establish independent task replication or "
                        "causation.",
    "matched_single": "Replication is absent: one failure-success pair only.",
    "cross_family_analogue": "An analogue across families, not a matched causal contrast; the "
                             "family and context differences are listed as confounders.",
    "policy_grounded_singleton": "No successful comparison exists; only a narrowly scoped "
                                 "procedural reminder grounded in the cited policy or tool "
                                 "contract is permitted.",
}
IDENTITY_LOOKUPS = frozenset({"find_user_id_by_email", "find_user_id_by_name_zip",
                              "get_user_details"})
IDENTIFICATION_CALLS = frozenset({"find_user_id_by_email", "find_user_id_by_name_zip"})
# Exact normative sentence of the public retail policy a consent singleton
# cites. The detector is disabled unless it occurs verbatim in policy.md.
CONSENT_RULE = ("Before taking any action that updates the database (cancel, modify, return, "
                "exchange), you must list the action details and obtain explicit user "
                "confirmation (yes) to proceed.")
# Amendment-3 slices keep the complete public dialogue (every user turn and
# every assistant text in full, every call and result); only these steps may
# omit anything, cumulatively in this order and only while the projected
# example exceeds TIERED_MAX_EXAMPLE_BYTES. No step removes dialogue text.
REDUCTION_RULE = "decisive-slice-v2"
REDUCTION_STEPS = ("public_think", "duplicate_heuristic_list", "repeated_identical_reads",
                   "unreferenced_lookups")
_YES = re.compile(r"\byes\b", re.IGNORECASE)


@dataclass(frozen=True)
class EvidenceRules:
    """Evidence and writer rules. The default is amendment 1/2 (two contrasts);
    ``amendment3()`` switches every amendment-3 rule on together."""

    tiers: bool = False
    slice_reduction: bool = False
    stage_prompt_budget: bool = False
    causal_critique: bool = False
    rendered_active_count: bool = False

    @classmethod
    def amendment3(cls) -> EvidenceRules:
        return cls(True, True, True, True, True)

    @property
    def name(self) -> str:
        if self == EvidenceRules.amendment3():
            return "amendment-3"
        if self == EvidenceRules():
            return "amendment-1"
        raise ValueError("evidence rules are registered as a whole (amendment-1 or amendment-3)")

    @classmethod
    def named(cls, name: str) -> EvidenceRules:
        if name == "amendment-3":
            return cls.amendment3()
        if name == "amendment-1":
            return cls()
        raise ValueError(f"unknown evidence rules {name!r}")


def _is_error_result(content: Any) -> bool:
    if isinstance(content, dict):
        return bool(content.get("error"))
    return isinstance(content, str) and content.lstrip().startswith("Error")


def full_slice(episode: Evidence) -> tuple[list[dict[str, Any]], list[int]]:
    """The complete public trajectory: every user turn and assistant text in
    full (never truncated), every call with its arguments and every result.
    Returns (slice, indices of empty messages left after think removal)."""
    messages = episode.messages
    first_user = next((i for i, m in enumerate(messages) if m["role"] == "user"), None)
    sliced, empty = [], []
    for i, message in enumerate(messages):
        calls = [c for c in message["tool_calls"] if c["name"] != "think"]
        if message["role"] == "assistant" and not calls and not message["content"]:
            empty.append(i)  # e.g. a turn whose only call was the private think tool
            continue
        if message["role"] == "user":
            stage = "request" if i == first_user else "user_turn"
        elif message["role"] == "tool":
            stage = "result"
        elif calls:
            stage = "action" if any(is_write(c["name"]) or
                                    c["name"] == "transfer_to_human_agents"
                                    for c in calls) else "lookup"
        else:
            stage = "assistant_text"
        sliced.append({"message_index": i, "stage": stage, "role": message["role"],
                       "content": message["content"],
                       "tool_calls": [{"name": c["name"], "arguments": c["arguments"]}
                                      for c in calls]})
    return sliced, empty


def _base_example(episode: Evidence, reference: str, operation: str) -> dict[str, Any]:
    if any(t.get("sanitized") is not True for t in episode.public_think):
        raise ValueError("writer thoughts require an audited sanitized public projection")
    sliced, empty = full_slice(episode)
    return {"reference": reference, "outcome": int(episode.reward == 1),
            "numeric_reward": episode.reward,
            "numeric_components": {k: v for k, v in episode.numeric_components.items()
                                   if k in COMPONENTS},
            "components": {k: int(v == 1) for k, v in episode.components.items()
                           if k in COMPONENTS},
            "termination": episode.termination_reason, "slice": sliced,
            "slice_coverage": {"total_messages": len(episode.messages),
                               "selected_messages": len(sliced),
                               "complete_dialogue": True,
                               "all_write_calls_retained": True,
                               "omitted_empty_messages": empty},
            "observed_signals": list(observed_signatures(episode, failures_only=False)),
            "heuristics_not_labels": list(heuristic_signatures(episode)),
            "heuristic_hypotheses": [{"signal": signal, "status": "hypothesis"}
                                     for signal in heuristic_signatures(episode)],
            "public_think_coverage": {"total": len(episode.public_think),
                                      "selected": min(MAX_THOUGHTS, len(episode.public_think))},
            "public_think": [{"turn": t.get("turn"), "index": t.get("index"),
                              "content": t["content"], "source": {
                                  "file": "think_public.jsonl", "reference": reference,
                                  "row": t.get("source_row")}, "sanitized": True}
                             for t in episode.public_think[:MAX_THOUGHTS]]}


def _measure(example: dict[str, Any]) -> int:
    example["projected_bytes"] = 0
    for _ in range(3):
        example["projected_bytes"] = len(json.dumps(example, separators=(",", ":"))
                                         .encode("utf-8"))
    return example["projected_bytes"]


def _aliases_in(value: Any) -> set[str]:
    return set(ALIAS.findall(value if isinstance(value, str) else json.dumps(value)))


def _omit(example: dict[str, Any], step: str, indices: list[int], **extra: Any) -> None:
    record = {"step": step, "message_indices": sorted(indices), **extra}
    example["slice_reduction"]["omitted"].append(record)
    keep = set(indices)
    example["slice"] = [m for m in example["slice"] if m["message_index"] not in keep]


def _cluster_op_action(message: dict[str, Any], operation: str) -> bool:
    return message["stage"] == "action" and any(
        call_operation(c["name"]) == operation for c in message["tool_calls"])


def _lookup_groups(sliced: list[dict[str, Any]]
                   ) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    groups = []
    for position, message in enumerate(sliced):
        if message["stage"] != "lookup":
            continue
        results = []
        for follower in sliced[position + 1:]:
            if follower["role"] != "tool":
                break
            results.append(follower)
        groups.append((message, results))
    return groups


def _droppable(message: dict[str, Any], results: list[dict[str, Any]],
               protected: set[int]) -> bool:
    """A lookup may be omitted only if it carries no dialogue text, no error
    result and no protected (decisive) message index."""
    indices = {message["message_index"], *(r["message_index"] for r in results)}
    return (not str(message["content"]).strip() and not indices & protected
            and not any(_is_error_result(r["content"]) for r in results)
            and not {c["name"] for c in message["tool_calls"]} & IDENTITY_LOOKUPS)


def _baselines(sliced: list[dict[str, Any]]) -> set[int]:
    return {m.get("baseline_message_index") for m in sliced} - {None}


def _step_repeated_identical_reads(example: dict[str, Any], protected: set[int]) -> bool:
    """Identical read calls with identical results and no dialogue text: keep
    the first and the last occurrence (the repetition stays visible)."""
    sliced = example["slice"]
    seen: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]]]]] = defaultdict(list)
    for message, results in _lookup_groups(sliced):
        key = json.dumps({"calls": message["tool_calls"],
                          "results": [r["content"] for r in results]}, sort_keys=True)
        seen[key].append((message, results))
    keep_baselines = _baselines(sliced)
    dropped = [x["message_index"] for groups in seen.values() for m, rs in groups[1:-1]
               if _droppable(m, rs, protected | keep_baselines) for x in [m, *rs]]
    if not dropped:
        return False
    _omit(example, "repeated_identical_reads", dropped,
          reason="identical read and result repeated; first and last kept")
    return True


def _step_unreferenced_lookups(example: dict[str, Any], operation: str,
                               protected: set[int]) -> bool:
    """Omit lookups (call and results, no dialogue text) that share no typed
    alias with any write action's arguments. Identity lookups, error results,
    protected decisive messages and delta baselines are always kept; without
    write actions nothing is omitted."""
    sliced = example["slice"]
    actions = [m for m in sliced if m["stage"] == "action"]
    if not actions:
        return False
    action_aliases = _aliases_in([c["arguments"] for m in actions for c in m["tool_calls"]])
    groups = _lookup_groups(sliced)
    drop = []
    for message, results in groups:
        shared = (_aliases_in([c["arguments"] for c in message["tool_calls"]]) |
                  _aliases_in([r["content"] for r in results])) & action_aliases
        if not shared and _droppable(message, results, protected):
            drop.append((message, results))
    while True:  # deltas need their baselines
        dropped_idx = {x["message_index"] for m, rs in drop for x in [m, *rs]}
        baselines = {m.get("baseline_message_index") for m in sliced
                     if m["message_index"] not in dropped_idx} - {None}
        keep = [(m, rs) for m, rs in drop if any(r["message_index"] in baselines for r in rs)]
        if not keep:
            break
        drop = [g for g in drop if g not in keep]
    if not drop:
        return False
    tools: dict[str, int] = defaultdict(int)
    for message, _ in drop:
        for call in message["tool_calls"]:
            tools[call["name"]] += 1
    _omit(example, "unreferenced_lookups",
          [x["message_index"] for m, rs in drop for x in [m, *rs]],
          tools=dict(sorted(tools.items())),
          reason="lookup without dialogue text that shares no typed identity with any "
                 "write action's arguments")
    return True


def reduce_example(example: dict[str, Any], operation: str, *,
                   limit: int = TIERED_MAX_EXAMPLE_BYTES, protected: set[int] | None = None,
                   trigger: str = "example_budget") -> dict[str, Any]:
    """Decisive-slice reduction (``decisive-slice-v2``) of a projected example.

    Applied only to an example over ``limit`` (the per-example budget), or in
    full (``limit`` 0) before a packet is refused for the total budget. Steps
    already applied are skipped. It never removes dialogue: no user turn and
    no assistant text (consent, conditions, refinements, proposals, replies)
    is omitted or truncated, nor any write with its arguments and result, any
    error result, any identity lookup, any delta baseline, or any ``protected``
    message (the evidence that makes the packet eligible at its tier). Every
    omission is recorded with its original message indices. An example still
    over the per-example budget is refused (``budget_omission``), never cut."""
    protected = set(protected or ())
    record = example.get("slice_reduction") or {
        "rule": REDUCTION_RULE, "original_projected_bytes": _measure(example),
        "steps": [], "omitted": []}
    example["slice_reduction"] = record
    applied_before = list(record["steps"])
    for step in REDUCTION_STEPS:
        if _measure(example) <= limit:
            break
        if step in record["steps"]:
            continue
        changed = False
        if step == "public_think" and example["public_think"]:
            count = len(example["public_think"])
            example["public_think"] = []
            example["public_think_coverage"]["selected"] = 0
            record["omitted"].append({"step": step, "thoughts": count,
                                      "reason": "optional self-reported think evidence"})
            changed = True
        elif step == "duplicate_heuristic_list" and "heuristics_not_labels" in example:
            example.pop("heuristics_not_labels")  # repeated in heuristic_hypotheses
            record["omitted"].append({"step": step, "reason": "duplicate of "
                                      "heuristic_hypotheses"})
            changed = True
        elif step == "repeated_identical_reads":
            changed = _step_repeated_identical_reads(example, protected)
        elif step == "unreferenced_lookups":
            changed = _step_unreferenced_lookups(example, operation, protected)
        if changed:
            record["steps"].append(step)
            if trigger != "example_budget":
                record.setdefault("total_budget_steps", []).append(step)
    if not record["steps"]:
        example.pop("slice_reduction")
    elif record["steps"] != applied_before:
        example["slice_coverage"]["selected_messages"] = len(example["slice"])
        example["slice_coverage"]["reduced"] = True
    _measure(example)
    return example


def _results_of(messages: tuple[dict[str, Any], ...], index: int) -> list[int]:
    found, j = [], index + 1
    while j < len(messages) and messages[j]["role"] == "tool":
        found.append(j)
        j += 1
    return found


def _json(text: Any) -> Any:
    try:
        return json.loads(text) if isinstance(text, str) else text
    except ValueError:
        return None


def _decision_context(episode: Evidence, operation: str) -> dict[str, Any]:
    """Comparable decision context of an episode for one operation (audited):
    the operation's write tools; the statuses of the ACTION'S TARGET order seen
    in lookup results before its first write; whether the user was
    SUCCESSFULLY identified (an identification call with a non-error result)
    before it; the policy prerequisite; and (reported as differences only) the
    lookup tools used and the termination. ``evidence`` holds the message
    indices of the target-status results and the identification results."""
    messages = episode.messages
    first = next((i for i, m in enumerate(messages) if any(
        call_operation(c["name"]) == operation for c in m["tool_calls"])), None)
    writes = sorted({c["name"] for m in messages for c in m["tool_calls"]
                     if call_operation(c["name"]) == operation})
    targets = set()
    if first is not None:
        targets = {str(c["arguments"].get("order_id")) for c in messages[first]["tool_calls"]
                   if call_operation(c["name"]) == operation and c["arguments"].get("order_id")}
    statuses, lookups, evidence_indices = set(), set(), []
    identified = False
    for i, message in enumerate(messages[:first if first is not None else 0]):
        names = {c["name"] for c in message["tool_calls"]}
        lookups.update(n for n in names if n != "think" and not call_operation(n))
        if message["role"] != "assistant" or not names:
            continue
        for j in _results_of(messages, i):
            result = _json(messages[j]["content"])
            if names & IDENTIFICATION_CALLS and not _is_error_result(
                    result if result is not None else messages[j]["content"]):
                identified = True
                evidence_indices += [i, j]
            if (isinstance(result, dict) and str(result.get("order_id")) in targets
                    and "status" in result):
                statuses.add(str(result["status"]))
                evidence_indices += [i, j]
    return {"write_tools": writes, "target_statuses_before_action": sorted(statuses),
            "user_identified": identified,
            "prerequisite": _PREREQUISITES.get(operation, "policy_prerequisites"),
            "lookup_tools": sorted(lookups), "termination": episode.termination_reason,
            "evidence": sorted(set(evidence_indices))}


CONTEXT_CRITERIA = ("write_tools", "target_statuses_before_action", "user_identified",
                    "prerequisite")


def comparable_context(failed: Evidence, successful: Evidence, operation: str
                       ) -> tuple[bool, list[str], list[str]]:
    """(comparable, satisfied criteria, material differences). Comparable only
    with non-empty, equal write tools and target-order status facts,
    successful identification in both and the same policy prerequisite."""
    a, b = _decision_context(failed, operation), _decision_context(successful, operation)
    satisfied = [f"same {key}: {a[key]}" for key in CONTEXT_CRITERIA if a[key] == b[key]]
    differences = [f"{key} differs: {a[key]} vs {b[key]}" for key in a
                   if key != "evidence" and a[key] != b[key]]
    ok = (all(a[key] == b[key] for key in CONTEXT_CRITERIA) and bool(a["write_tools"])
          and bool(a["target_statuses_before_action"]) and a["user_identified"] is True)
    return ok, satisfied, differences


@functools.lru_cache(maxsize=4)
def contract_preconditions(path: str = str(TOOLS)) -> dict[str, list[str]]:
    """Per public write tool: the precondition-violation messages its own body
    raises (``raise ValueError("...")`` literals; formatted parts become
    wildcards). Helper-raised lookups such as "Order not found" are excluded."""
    source = public_path(Path(path)).read_text(encoding="utf-8")
    found: dict[str, list[str]] = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef) or node.name not in _WRITE_OPERATIONS:
            continue
        patterns = []
        for raised in ast.walk(node):
            if not (isinstance(raised, ast.Raise) and isinstance(raised.exc, ast.Call)
                    and raised.exc.args):
                continue
            arg = raised.exc.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                patterns.append(re.escape(arg.value))
            elif isinstance(arg, ast.JoinedStr):
                patterns.append("".join(re.escape(v.value) if isinstance(v, ast.Constant)
                                        else ".+" for v in arg.values))
        found[node.name] = patterns
    return found


def singleton_discrepancies(example: dict[str, Any], operation: str, policy_text: str,
                            contracts: dict[str, str], tools_path: Path = TOOLS
                            ) -> list[dict[str, Any]]:
    """Visible, operation-specific violated preconditions in one complete
    failure slice (priority order):

    - ``write_without_explicit_yes``: a write of the operation before which NO
      user turn in the complete dialogue contains an explicit "yes" (the
      cited consent sentence);
    - ``write_precondition_error``: a write of the operation whose error result
      is one of the precondition violations its own contract body raises
      (transient or service errors never match).

    Nothing is inferred from an incomplete slice: the slice must carry the
    complete dialogue."""
    sliced = example["slice"]
    if not example.get("slice_coverage", {}).get("complete_dialogue"):
        return []
    found = []
    if CONSENT_RULE in policy_text and operation not in ("transfer", "information"):
        for position, message in enumerate(sliced):
            if not _cluster_op_action(message, operation):
                continue
            users = [m for m in sliced[:position] if m["role"] == "user"]
            if users and not any(_YES.search(str(m["content"])) for m in users):
                found.append({"kind": "write_without_explicit_yes",
                              "message_indices": [*(m["message_index"] for m in users),
                                                  message["message_index"]],
                              "visible": "no user turn before this write contains an "
                                         "explicit 'yes'",
                              "normative": {"source": "policy", "text": CONSENT_RULE}})
                break
    preconditions = contract_preconditions(str(tools_path))
    for position, message in enumerate(sliced):
        if not _cluster_op_action(message, operation):
            continue
        name = next(c["name"] for c in message["tool_calls"]
                    if call_operation(c["name"]) == operation)
        for result in sliced[position + 1:]:
            if result["role"] != "tool":
                break
            text = str(result["content"]).strip()
            text = text[len("Error:"):].strip() if text.startswith("Error:") else None
            if text and name in contracts and any(
                    re.fullmatch(p, text) for p in preconditions.get(name, [])):
                found.append({"kind": "write_precondition_error",
                              "message_indices": [message["message_index"],
                                                  result["message_index"]],
                              "visible": f"the {name} call failed a precondition its "
                                         f"contract raises: {text}",
                              "normative": {"source": "tool_contract", "tool": name,
                                            "text": contracts[name]}})
                return found
    return found


def _key(episode: Evidence) -> tuple[str, str]:
    return (episode.run_manifest, episode.episode_id)


SINGLETON_ORDER = ("write_without_explicit_yes", "write_precondition_error")


def select_tiered(cluster: BehaviouralCluster, usable: set[tuple[str, str]] | None,
                  examples: dict[tuple[str, str], dict[str, Any]], policy_text: str,
                  contracts: dict[str, str], tools_path: Path = TOOLS
                  ) -> dict[str, Any] | None:
    """The strongest tier available from ``usable`` episodes (all if None),
    judged on ``examples`` (complete, unreduced slices for eligibility)."""
    def ok(episode: Evidence) -> bool:
        return usable is None or _key(episode) in usable

    failures = [e for e in cluster.failures if ok(e)]
    successes = [e for e in cluster.episodes if e.reward == 1 and ok(e)]
    sizes = {_key(e): examples[_key(e)]["projected_bytes"] for e in cluster.episodes}
    success_tasks = {s.task_id for s in successes}
    failures.sort(key=lambda e: (e.task_id not in success_tasks, sizes[_key(e)]))
    successes.sort(key=lambda e: sizes[_key(e)])
    pairs: list[tuple[Evidence, Evidence, str]] = []
    used_f: set[tuple[str, str]] = set()
    used_s: set[tuple[str, str]] = set()
    for match in ("same_task", "same_family"):  # no episode reused across pairs
        for failed in failures:
            if _key(failed) in used_f or len(pairs) == 2:
                continue
            successful = next((s for s in successes if _key(s) not in used_s and (
                s.task_id == failed.task_id if match == "same_task" else
                failed.family is not None and failed.family == s.family)), None)
            if successful is not None:
                pairs.append((failed, successful, match))
                used_f.add(_key(failed))
                used_s.add(_key(successful))
    tier, singleton = None, None
    if pairs:
        tier = "matched_repeated" if len(pairs) == 2 else "matched_single"
    else:
        analogues = []
        for failed in failures:
            for successful in successes:
                if failed.family is None or successful.family is None or \
                        failed.family == successful.family:
                    continue
                comparable, _, differences = comparable_context(failed, successful,
                                                                cluster.operation)
                if comparable:
                    analogues.append((len(differences), sizes[_key(failed)]
                                      + sizes[_key(successful)], failed, successful))
        if analogues:
            analogues.sort(key=lambda a: (a[0], a[1], _key(a[2]), _key(a[3])))
            _, _, failed, successful = analogues[0]
            pairs = [(failed, successful, "cross_family_analogue")]
            used_f.add(_key(failed))
            used_s.add(_key(successful))
            tier = "cross_family_analogue"
        else:
            ranked = []
            for failed in failures:
                hits = singleton_discrepancies(examples[_key(failed)], cluster.operation,
                                               policy_text, contracts, tools_path)
                if hits:
                    order = SINGLETON_ORDER.index(hits[0]["kind"])
                    ranked.append((order, sizes[_key(failed)], _key(failed), failed, hits[0]))
            if ranked:
                ranked.sort(key=lambda r: r[:3])
                singleton = (ranked[0][3], ranked[0][4])
                used_f.add(_key(singleton[0]))
                tier = "policy_grounded_singleton"
    if tier is None:
        return None
    all_success_tasks = {s.task_id for s in cluster.episodes if s.reward == 1}
    # As in the two-contrast rule: never-passing failures may be the paired ones.
    never_pass = [e for e in failures if e.task_id not in (
        cluster.passing_tasks | all_success_tasks)][:2]
    controls = [s for s in successes if _key(s) not in used_s][
        :2 if tier != "policy_grounded_singleton" else 1]
    if not controls and tier != "policy_grounded_singleton":
        controls = [p[1] for p in pairs][:1]
    return {"tier": tier, "pairs": pairs, "singleton": singleton,
            "never_pass": never_pass, "controls": controls}


def decisive_indices(selection: dict[str, Any], operation: str
                     ) -> dict[tuple[str, str], set[int]]:
    """Per episode: the message indices that make the selection eligible at
    its tier (a singleton's discrepancy; an analogue's target-status and
    identification facts). Reduction must keep them all."""
    found: dict[tuple[str, str], set[int]] = defaultdict(set)
    for failed, successful, match in selection["pairs"]:
        if match == "cross_family_analogue":
            for episode in (failed, successful):
                found[_key(episode)] |= set(_decision_context(episode, operation)["evidence"])
    if selection["singleton"] is not None:
        failed, hit = selection["singleton"]
        found[_key(failed)] |= set(hit["message_indices"])
    return found


def tiered_bundle(cluster: BehaviouralCluster, notes: list[dict[str, Any]], *,
                  policy_path: Path = POLICY, tools_path: Path = TOOLS,
                  provenance: dict[str, Any] | None = None,
                  reduce: bool = True,
                  floor: list[tuple[Evidence, dict[str, Any]]] | None = None
                  ) -> dict[str, Any]:
    """Amendment-3 packet: tier, labels, reduced decisive slices, audit map.

    ``floor`` (amendment 6b; None keeps the amendment-3 packet byte for byte):
    the failure episodes that meet the two-family discrepancy floor, each with
    its visible discrepancy. They are added as references (role ``discrepancy
    failure (floor)``), their discrepancy messages are protected from
    reduction, and the floor (not the tier) decides sufficiency; the tier and
    its comparisons stay as context."""
    policy = policy_excerpt(cluster.operation, policy_path)
    policy_text = public_path(policy_path).read_text(encoding="utf-8")
    all_names = {c["name"] for e in cluster.episodes for m in e.messages
                 for c in m["tool_calls"] if c["name"] != "think"}
    contracts = {c["tool"]: c["contract"] for c in tool_contracts(all_names, tools_path)}
    # Eligibility and tier come from the COMPLETE, unreduced slices; reduction
    # can never create, remove or change them.
    full: dict[tuple[str, str], dict[str, Any]] = {}
    for episode in cluster.episodes:
        example = _base_example(episode, "E0", cluster.operation)
        _project_example(episode, example)
        _measure(example)
        full[_key(episode)] = example
    strongest = select_tiered(cluster, None, full, policy_text, contracts, tools_path)
    examples = full
    fitting = None
    floor_protected: dict[tuple[str, str], set[int]] = defaultdict(set)
    for failed, hit in floor or []:
        floor_protected[_key(failed)] |= set(hit["message_indices"])
    if floor and strongest is None:
        # The floor alone qualifies the packet: reduce with its messages protected.
        examples = {}
        for key, example in full.items():
            reduced = json.loads(json.dumps(example))
            if reduced["projected_bytes"] > TIERED_MAX_EXAMPLE_BYTES and reduce:
                reduce_example(reduced, cluster.operation, protected=floor_protected.get(key))
            examples[key] = reduced
    if strongest is not None:
        protected = decisive_indices(strongest, cluster.operation)
        for key, indices in floor_protected.items():
            protected[key] |= indices
        examples = {}
        for key, example in full.items():
            reduced = json.loads(json.dumps(example))
            if reduced["projected_bytes"] > TIERED_MAX_EXAMPLE_BYTES and reduce:
                reduce_example(reduced, cluster.operation, protected=protected.get(key))
            examples[key] = reduced
        fits = {k for k, e in examples.items() if e["projected_bytes"] <= TIERED_MAX_EXAMPLE_BYTES}
        # Within the SAME tier, prefer comparisons whose examples fit; judged on
        # the complete slices, then verified to keep their decisive messages.
        fitting = select_tiered(cluster, fits, full, policy_text, contracts, tools_path)
        if fitting is not None:
            keep = decisive_indices(fitting, cluster.operation)
            if any(not indices <= {m["message_index"] for m in examples[key]["slice"]}
                   for key, indices in keep.items()):
                fitting = None
    status, reason, chosen = "sufficient", None, fitting
    if floor and (strongest is None or fitting is None or fitting["tier"] != strongest["tier"]):
        # Amendment 6b: the tier is context only; the floor decides sufficiency.
        chosen = fitting if fitting is not None and strongest is not None and \
            fitting["tier"] == strongest["tier"] else None
    elif strongest is None:
        status = "insufficient_evidence"
        reason = ("No evidence tier applies: no matched or comparable cross-family "
                  "failure-success contrast and no visible operation-specific violated "
                  "precondition against the cited policy sentence or tool contract.")
        chosen = None
    elif fitting is None or fitting["tier"] != strongest["tier"]:
        # Size pressure never relabels evidence: no downgrade, no clipping.
        status = "too_large"
        reason = (f"The strongest available tier ({strongest['tier']}) needs an example that "
                  "exceeds the per-example byte budget after decisive-slice reduction.")
        chosen = strongest
    references: dict[tuple[str, str], str] = {}
    local: list[Evidence] = []
    roles: dict[str, list[str]] = defaultdict(list)

    def reference(episode: Evidence, role: str) -> str:
        if _key(episode) not in references:
            references[_key(episode)] = f"E{len(references) + 1}"
            local.append(episode)
        ref = references[_key(episode)]
        if role not in roles[ref]:
            roles[ref].append(role)
        return ref

    contrasts, match_criteria, singleton_refs = [], [], []
    missing: list[str] = []
    confounders = ["Rewards and component scores are outcomes, not causal labels.",
                   "Heuristic flags and the prerequisite scaffold are unverified hypotheses."]
    if chosen is not None:
        for number, (failed, successful, match) in enumerate(chosen["pairs"], start=1):
            comparison = f"C{number}"
            f_ref = reference(failed, f"failure ({comparison})")
            s_ref = reference(successful, f"success ({comparison})")
            contrasts.append({"comparison": comparison, "match": match, "failed": f_ref,
                              "successful": s_ref})
            _, satisfied, differences = comparable_context(failed, successful,
                                                           cluster.operation)
            criteria = [f"same operation: {cluster.operation}",
                        f"same policy prerequisite: {cluster.prerequisite}"]
            if match == "same_task":
                criteria.append("same task")
            elif match == "same_family":
                criteria.append("same task family (different task)")
                missing.append(f"{comparison}: no same-task success; matched within family")
                confounders.append(f"{comparison}: different tasks within one family.")
            else:
                criteria += satisfied
                confounders.append(f"{comparison}: different task families (analogue).")
            if differences:
                confounders.append(f"{comparison}: " + "; ".join(differences))
            match_criteria.append({"comparison": comparison, "match": match,
                                   "failed": f_ref, "successful": s_ref,
                                   "satisfied": criteria})
        if chosen["singleton"] is not None:
            failed, hit = chosen["singleton"]
            ref = reference(failed, "singleton failure")
            singleton_refs.append(ref)
            match_criteria.append({"singleton": ref, "discrepancy": {
                "kind": hit["kind"], "visible": hit["visible"],
                "message_indices": hit["message_indices"]},
                "normative_support": hit["normative"]})
            missing.append("No successful comparison: no same-task, same-family or comparable "
                           "cross-family success.")
        if chosen["tier"] in ("matched_single", "cross_family_analogue",
                              "policy_grounded_singleton"):
            missing.append("Replication: no second distinct failure-success pair.")
        if chosen["tier"] == "cross_family_analogue":
            missing.append("No same-task or same-family success.")
        never = [reference(e, "never-passing failure") for e in chosen["never_pass"]]
        negative = [reference(e, "successful counterexample") for e in chosen["controls"]]
        if chosen["tier"] == "policy_grounded_singleton" and negative:
            confounders.append("Successful counterexamples of this operation are not "
                               "comparable decision contexts.")
    else:
        never, negative = [], []
        if not floor:
            for episode in cluster.failures[:2]:
                reference(episode, "failure")
        else:
            # Amendment 6b: successes of the operation stay as counterexamples.
            successes = sorted((e for e in cluster.episodes if e.reward == 1),
                               key=lambda e: (examples[_key(e)]["projected_bytes"], _key(e)))
            negative = [reference(e, "successful counterexample") for e in successes[:2]]
    floor_refs = []
    for failed, hit in floor or []:
        ref = reference(failed, "discrepancy failure (floor)")
        floor_refs.append(ref)
        match_criteria.append({"floor": ref, "discrepancy": {
            "kind": hit["kind"], "visible": hit["visible"],
            "message_indices": hit["message_indices"]},
            "normative_support": hit["normative"]})
    if provenance is not None:
        for episode in local:
            ref = references[_key(episode)]
            provenance[ref] = {
                "run_manifest": episode.run_manifest, "episode_id": episode.episode_id,
                "task_id": episode.task_id, "benchmark_family": episode.family,
                "roles": roles[ref], "source_sha256": hashlib.sha256(json.dumps(
                    list(episode.messages), sort_keys=True).encode()).hexdigest()}
    examples_out, names = [], set()
    decisive = decisive_indices(chosen, cluster.operation) if chosen is not None else {}
    for key, indices in floor_protected.items():
        decisive = dict(decisive)
        decisive[key] = set(decisive.get(key, set())) | indices
    for episode in local:
        example = json.loads(json.dumps(examples[_key(episode)]))
        ref = references[_key(episode)]
        example["reference"] = ref
        example["role"] = roles[ref]
        if decisive.get(_key(episode)):
            # The messages that make this packet eligible at its tier; never
            # omitted by any later reduction.
            example["decisive_message_indices"] = sorted(decisive[_key(episode)])
        for thought in example["public_think"]:
            thought["source"]["reference"] = ref
        names.update(c["name"] for m in example["slice"] for c in m["tool_calls"])
        _measure(example)
        if example["projected_bytes"] > TIERED_MAX_EXAMPLE_BYTES:
            example["slice"] = []
            example["public_think"] = []
            example["budget_omission"] = True
            example.pop("slice_reduction", None)
            _measure(example)
        elif example.get("slice_reduction"):
            omitted = sum(len(o.get("message_indices", [])) for o in
                          example["slice_reduction"]["omitted"])
            confounders.append(f"{ref}: decisive-slice reduction omitted {omitted} messages "
                               "(see slice_reduction); absence there is not evidence.")
        if provenance is not None:
            provenance[ref]["message_indices"] = [m["message_index"] for m in example["slice"]]
            provenance[ref]["omitted"] = example.get("slice_reduction", {}).get("omitted", [])
        examples_out.append(example)
    contracts_out = tool_contracts(names, tools_path)
    refs_out = [{"ref": references[_key(e)], "kind": "example",
                 "role": roles[references[_key(e)]]} for e in local]
    refs_out.append({"ref": "P1", "kind": "policy", "source": policy["source"],
                     "sha256": policy["sha256"], "cites": "policy excerpt in this packet"})
    for criterion in match_criteria:
        support = criterion.get("normative_support")
        if support:
            entry = {"ref": "P2" if support["source"] == "policy" else "T1",
                     "kind": support["source"], "text": support["text"],
                     **({"tool": support["tool"]} if "tool" in support else {})}
            if entry not in refs_out:
                refs_out.append(entry)
    applicable = [{"content": n["content"]} for n in notes if not n.get("target_signature") or
                  n["target_signature"] in {cluster.target_signature, cluster.id} or
                  n.get("scope") == "global"]
    tier = None if chosen is None else chosen["tier"]
    payload = {"cluster_id": cluster.id, "operation": cluster.operation,
               "prerequisite": cluster.prerequisite, "observable_mechanism": cluster.mechanism,
               "target_signature": cluster.target_signature,
               "mixed_stream_tasks": cluster.mixed_tasks,
               "evidence_tier": tier,
               "tier_limitation": None if tier is None else TIER_LIMITATIONS[tier],
               "references": refs_out, "match_criteria": match_criteria,
               "missing_comparisons": missing, "confounders": confounders,
               "mechanism_status": "hypothesis",
               "disclosures": {
                   "prerequisite_labels": "Human-written _PREREQUISITES; unverified policy scaffold, not causal labels.",
                   "operation_inference": "Called write tools; request heuristic only if none called.",
                   "heuristic_flags": "Unverified hypotheses; never cluster partitions.",
                   "lookup_projection": "Used/discussed variants + sample. Orders omit unrelated nulls, fulfillments and addresses (except address writes); ownership/items/payments/action fields kept. Deltas inherit absent fields from baseline_message_index; removed fields listed.",
                   "dialogue": "Complete public dialogue: every user turn and every assistant text in full (never truncated), every call with its arguments and every result; the private think tool is excluded (empty turns listed in slice_coverage).",
                   "slice_reduction": f"{REDUCTION_RULE}: oversized examples only, steps {list(REDUCTION_STEPS)} in order; no dialogue text, no write with its arguments and result, no error result, no identity lookup and no decisive message is ever removed; omissions listed with original message indices.",
                   "tool_result_encoding": "JSON results embedded as objects/lists; other results as text.",
                   "example_byte_limit": TIERED_MAX_EXAMPLE_BYTES,
                   "thought_limits": {"count": MAX_THOUGHTS, "characters_each": MAX_THOUGHT_CHARS}},
               "evidence_status": status, "evidence_reason": reason,
               "alias_scope": "Per-example aliases; equal aliases across references do not imply equal identities",
               "evidence_hierarchy": [
                   {"kind": "public_policy_and_tool_contracts", "use": "normative support"},
                   {"kind": "visible_messages_calls_results", "use": "observable facts"},
                   {"kind": "numeric_stream_outcomes", "use": "outcomes, never causal labels"},
                   {"kind": "sanitized_public_think", "use": "agent hypothesis, not verified fact"},
                   {"kind": "heuristics", "use": "marked hypotheses, never training labels or cluster partitions"}],
               "pairs": contrasts, "singleton_failures": singleton_refs,
               "never_pass_examples": never, "successful_negative_controls": negative,
               "examples": examples_out, "policy": policy,
               "tool_contracts": contracts_out, "current_notes": applicable,
               "finding_chain": {
                   "observable_fact": [{"reference": e["reference"], "components": e["components"],
                                        "signals": e["observed_signals"]} for e in examples_out],
                   "suspected_mechanism": {"status": "hypothesis", "text": "Unresolved; infer from public evidence and policy only."},
                   "proposed_intervention": None,
                   "measured_effect": {"status": "not_measured", "value": None}}}
    if floor:
        payload["discrepancy_floor"] = floor_refs
    if status == "sufficient" and any(e.get("budget_omission") for e in examples_out
                                      if set(e["role"]) & {"failure (C1)", "success (C1)",
                                                           "failure (C2)", "success (C2)",
                                                           "singleton failure",
                                                           "discrepancy failure (floor)"}):
        payload["evidence_status"] = "too_large"
        payload["evidence_reason"] = "Required example exceeds per-example byte budget."
    check_no_grading_fields(payload)
    payload["missing_tool_contracts"] = sorted(names - {c["tool"] for c in contracts_out})
    if contracts_out:
        payload["tool_contracts_source"] = {key: contracts_out[0][key]
                                            for key in ("source", "sha256")}
        payload["tool_contracts"] = [{k: v for k, v in c.items() if k not in {"source", "sha256"}}
                                     for c in contracts_out]
    return payload


def tier_rank(bundle: dict[str, Any]) -> int:
    tier = bundle.get("evidence_tier")
    return TIER_RANK.get(tier, len(TIERS)) if bundle.get("evidence_status") == "sufficient" \
        else len(TIERS) + 1


def bundle_references(bundle: dict[str, Any]) -> set[str]:
    return {e["reference"] for e in bundle["examples"]}
