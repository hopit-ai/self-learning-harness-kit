"""Offline pruning baselines for OpenAI-format Hermes message snapshots.

Only tool-result bodies are masked. Every other field, message, and call/result
edge is preserved. All arms use the same conservative protection gate. A finite
lexical detector cannot prove arbitrary prose safe: integrate any host-specific
protection rules with ``protected_detector`` (combined with, never replacing, the
built-in detector), and use ``protected=True`` on known sensitive tool messages.

These are baseline implementations inspired by recency masking, DTOC and LRE,
not reproductions of their published code. Features see only the supplied
boundary snapshot, never a replay continuation or its labels.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .probe_storage import ContentStore, canonical_bytes, content_reference

Message = dict[str, Any]
Detector = Callable[[Mapping[str, Any]], bool]

# Conservative, intentionally favors retaining a false positive. Structured keys
# are normalized separately, so orderId, user_id, shipping-address all match.
_SENSITIVE_KEY = re.compile(
    r"(?:id|ids|uuid|guid|identifier|identifiers|address|addresses|email|phone|"
    r"postalcode|zipcode|postcode|sku|ssn|ordernumber|accountnumber|trackingnumber|"
    r"street|city|state|country|zip|apartment|unit|house|contact|"
    r"order|user|customer|account|consent|confirm|confirmation|confirmed|"
    r"permission|authorization|authorisation|policy|policies|constraint|constraints|"
    r"instruction|instructions|rule|rules|requirement|requirements|protected)", re.IGNORECASE)
_PROTECTED_TEXT = re.compile(
    r"\b(?:consent\w*|confirm\w*|approv\w*|authori[sz]\w*|permission\w*|"
    r"policy|policies|constraint\w*|must|shall|should|never|cannot|can't|"
    r"do\s+not|don't|only|required?|requirements?|prohibit\w*|forbid\w*|"
    r"allowed?|disallowed?|restrict\w*|instructions?|rules?|within|unless|except|"
    r"minimum|maximum|deadline|eligible|ineligible|permitted|forbidden|"
    r"yes|no|okay|ok|agreed|proceed|go\s+ahead|sounds\s+good|please)\b|"
    r"\b(?:order|user|customer|account|item|product|ticket|reservation|booking|"
    r"transaction|payment|return|tracking|reference|ids?|identifiers?|sku|ssn|"
    r"address|email|phone|postcode|"
    r"zip\s*code)\b|"
    r"\b[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9.-]+\.[a-z]{2,}\b|"
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|"
    r"\b\d+\s+[\w .'-]+\s(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|"
    r"boulevard|blvd|way|court|ct|terrace|place|pl|loop|circle|cir)\b|"
    r"\b\d{4,}\b|\b(?=[a-z0-9_-]*[a-z])(?=[a-z0-9_-]*\d)[a-z0-9_-]{3,}\b",
    re.IGNORECASE,
)
_PROTECTED_TOOL = re.compile(
    r"\b(?:policy|constraint|instructions?|consent|confirm\w*|rules?)\b", re.IGNORECASE)


def _protected_tool(name: str) -> bool:
    return bool(_PROTECTED_TOOL.search(name.replace("_", " ")))


def _sensitive(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            # Suffix catches arbitrary resource IDs, e.g. warehouse_id / orderId.
            if (_SENSITIVE_KEY.fullmatch(normalized) or normalized.endswith(("id", "ids"))
                    or any(term in normalized for term in
                           ("address", "consent", "confirm", "approv", "authoriz", "authoris",
                            "permission", "policy", "constraint", "instruction", "requirement"))):
                return True
            if _sensitive(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_sensitive(item) for item in value)
    if isinstance(value, str):
        # JSON structured bodies are common in Hermes. Inspect keys as well as prose.
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            decoded = None
        if isinstance(decoded, (dict, list)) and _sensitive(decoded):
            return True
        return bool(_PROTECTED_TEXT.search(value))
    # Bare large numeric identifiers are indistinguishable from measurements.
    return isinstance(value, (int, float)) and not isinstance(value, bool) and abs(value) >= 1000


def is_protected_content(message: Mapping[str, Any]) -> bool:
    """Conservatively protect turns, explicit metadata, structured IDs and prose.

    Non-tool turns are always protected. For tools, metadata other than standard
    transport fields is inspected too; tool_call_id is transport, not result text.
    On ambiguous or unstructured (non-text) bodies, keep the result unchanged.
    """
    if message.get("role") != "tool" or message.get("protected"):
        return True
    if _protected_tool(str(message.get("name", ""))):
        return True
    for key, value in message.items():
        if (key not in {"role", "tool_call_id", "name", "content", "protected"}
                and _sensitive({key: value})):
            return True
    body = message.get("content")
    if not isinstance(body, (str, dict, list)):
        return True
    # Multimodal/opaque content cannot safely be understood by this lexical gate.
    if isinstance(body, list) and any(not isinstance(p, str) for p in body):
        return True
    return _sensitive(body)


def validate_pairing(messages: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int, int, str]]:
    """Validate complete parallel-call groups, returning id -> (call, result, name).

    Invalid/incomplete snapshots raise ValueError before pruning. Call IDs must be
    unique in a snapshot, and each assistant group must be followed by its results.
    """
    pending: dict[str, tuple[int, str]] = {}
    seen: set[str] = set()
    pairs = {}
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError("messages must be mappings")
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                raise ValueError("orphan, duplicate, or out-of-order tool result")
            call_index, name = pending.pop(call_id)
            pairs[call_id] = (call_index, index, name)
            continue
        if pending:
            raise ValueError("tool call group is missing results")
        calls = message.get("tool_calls") or []
        if not isinstance(calls, (list, tuple)):
            raise ValueError("tool_calls must be a sequence of call mappings")
        if calls and message.get("role") != "assistant":
            raise ValueError("tool calls must belong to assistant turns")
        for call in calls:
            if not isinstance(call, Mapping):
                raise ValueError("tool calls must be mappings")
            call_id = call.get("id")
            function = call.get("function")
            if (not isinstance(call_id, str) or not call_id or call_id in seen
                    or not isinstance(function, Mapping)
                    or not isinstance(function.get("name"), str) or not function["name"]):
                raise ValueError("tool calls need unique IDs and function names")
            seen.add(call_id)
            pending[call_id] = (index, function["name"])
    if pending:
        raise ValueError("tool call group is missing results")
    return pairs


@dataclass
class PruneResult:
    messages: list[Message]
    # Text-free metadata only. Reversible bodies belong in the caller's ContentStore.
    masked: list[dict[str, Any]] = field(default_factory=list)

    @property
    def pruned_call_ids(self) -> list[str]:
        """Local integration handle; hash these IDs before exporting artifacts."""
        indices = {int(item["index"]) for item in self.masked}
        return [m["tool_call_id"] for i, m in enumerate(self.messages) if i in indices]


def _mask(messages: Sequence[Mapping[str, Any]], keep_indices: set[int], *,
          store: ContentStore | None = None, protected_detector: Detector | None = None,
          eligible_tools: Sequence[str] | None = None) -> PruneResult:
    pairs = validate_pairing(messages)
    result = PruneResult(deepcopy([dict(m) for m in messages]))
    for call_id, (call_index, index, name) in pairs.items():
        message = messages[index]
        # The accompanying assistant's prose can carry a confirmation/constraint.
        call_content = messages[call_index].get("content")
        if (index in keep_indices or is_protected_content(message)
                or (call_content and _sensitive(call_content))
                or _protected_tool(name)
                or (protected_detector is not None and protected_detector(message))
                or (eligible_tools is not None and name not in eligible_tools)):
            continue
        body = message.get("content")
        reference = content_reference(body)
        ref = str(reference["ref"])
        if store is not None:
            stub = f"[Result masked; restore {ref}]"
        else:
            stub = "[Earlier tool result masked]"
        # A pruning arm should not inflate short results with a longer stub.
        if len(canonical_bytes(stub)) >= len(canonical_bytes(body)):
            continue
        if store is not None:
            store.put(body)
        result.messages[index]["content"] = stub
        result.masked.append({"index": index, "call_ref": content_reference(call_id)["ref"],
                              **reference, "reversible": int(store is not None)})
    return result


def keep_last_k(messages: Sequence[Mapping[str, Any]], k: int = 5, *,
                protected_detector: Detector | None = None,
                eligible_tools: Sequence[str] | None = None) -> PruneResult:
    """Keep last k tool bodies, plus all protected results; mask older bodies.

    k counts *all* tool results (including protected ones), not assistant groups.
    Call/result skeletons survive even for masked results.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    validate_pairing(messages)
    indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    keep = set(indices[-k:]) if k else set()
    return _mask(messages, keep, protected_detector=protected_detector,
                 eligible_tools=eligible_tools)


def reversible_mask(messages: Sequence[Mapping[str, Any]], store: ContentStore, *,
                    k: int = 5, protected_detector: Detector | None = None,
                    eligible_tools: Sequence[str] | None = None) -> PruneResult:
    """Recency baseline with content-addressed restoration, retaining last k bodies."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    validate_pairing(messages)
    indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    return _mask(messages, set(indices[-k:]) if k else set(), store=store,
                 protected_detector=protected_detector, eligible_tools=eligible_tools)


def restore_masked(result: PruneResult, store: ContentStore) -> list[Message]:
    """Restore reversible bodies exactly; reject irreversible or mismatched data."""
    restored = deepcopy(result.messages)
    for item in result.masked:
        if not item["reversible"]:
            raise ValueError("irreversible recency mask cannot be restored")
        index = int(item["index"])
        if content_reference(restored[index]["tool_call_id"])["ref"] != item["call_ref"]:
            raise ValueError("masked result no longer matches its call")
        if restored[index].get("content") != f"[Result masked; restore {item['ref']}]":
            raise ValueError("masked placeholder was changed; refusing to overwrite it")
        body = store.get(str(item["ref"]))
        if content_reference(body)["sha256"] != item["sha256"]:
            raise ValueError("stored result digest mismatch")
        restored[index]["content"] = body
    validate_pairing(restored)
    return restored


def no_additional_pruning(messages: Sequence[Mapping[str, Any]]) -> PruneResult:
    validate_pairing(messages)
    return PruneResult(deepcopy([dict(m) for m in messages]))


@dataclass(frozen=True)
class ToolResultFeatures:
    age: int
    size: int  # UTF-8 serialized body bytes
    tool_name: str
    referenced_later_overlap: float  # Jaccard overlap, snapshot-local only


def extract_features(messages: Sequence[Mapping[str, Any]]) -> dict[int, ToolResultFeatures]:
    """Features indexed by result position; no protected text enters a fitted model."""
    pairs = validate_pairing(messages)
    ordered = sorted((result, name) for _, result, name in pairs.values())
    features = {}

    def words(value: Any) -> set[str]:
        if value is None:
            return set()
        text = value if isinstance(value, str) else canonical_bytes(value).decode()
        return set(re.findall(r"\b\w+\b", text.lower()))

    for rank, (index, name) in enumerate(ordered):
        body = messages[index].get("content")
        body_words = words(body)
        later = set().union(*(words(m.get("content")) for m in messages[index + 1:]))
        union = body_words | later
        features[index] = ToolResultFeatures(len(ordered) - rank - 1,
                                             len(canonical_bytes(body)), name,
                                             len(body_words & later) / len(union) if union else 0.0)
    return features


class FeatureScorer:
    """Deterministic, dependency-free logistic keep-probability baseline.

    Inputs are features only, labels are 0=mask / 1=keep. Numeric features are
    log-scaled age/size and lexical overlap; tool names use a fitted one-hot
    vocabulary. Tune thresholds on calibration data outside this class.
    """

    def __init__(self) -> None:
        self.tool_names: tuple[str, ...] = ()
        self.weights: list[float] = []

    def _vector(self, feature: ToolResultFeatures, *,
                tool_names: Sequence[str] | None = None) -> list[float]:
        if (not math.isfinite(feature.age) or not math.isfinite(feature.size)
                or feature.age < 0 or feature.size < 0
                or not math.isfinite(feature.referenced_later_overlap)
                or not 0 <= feature.referenced_later_overlap <= 1):
            raise ValueError("invalid result features")
        return [1.0, math.log1p(feature.age) / 10, math.log1p(feature.size) / 10,
                feature.referenced_later_overlap,
                *(float(feature.tool_name == name) for name in
                  (self.tool_names if tool_names is None else tool_names))]

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1 / (1 + math.exp(-max(-40.0, min(40.0, value))))

    def fit(self, features: Sequence[ToolResultFeatures], labels: Sequence[int], *,
            iterations: int = 500, learning_rate: float = 0.2, l2: float = 0.01
            ) -> FeatureScorer:
        if not features or len(features) != len(labels) or any(y not in (0, 1) for y in labels):
            raise ValueError("fit needs aligned nonempty features and binary labels")
        if (isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0
                or not math.isfinite(learning_rate) or learning_rate <= 0
                or not math.isfinite(l2) or l2 < 0):
            raise ValueError("invalid fitting parameters")
        tool_names = tuple(sorted({f.tool_name for f in features}))
        rows = [self._vector(f, tool_names=tool_names) for f in features]
        weights = [0.0] * len(rows[0])
        for _ in range(iterations):
            gradient = [0.0] * len(weights)
            for row, label in zip(rows, labels):
                error = self._sigmoid(sum(w * x for w, x in zip(weights, row))) - label
                for j, value in enumerate(row):
                    gradient[j] += error * value / len(rows)
            for j in range(len(weights)):
                weights[j] -= learning_rate * (gradient[j] + (l2 * weights[j] if j else 0))
        self.tool_names, self.weights = tool_names, weights
        return self

    def predict_proba(self, features: Sequence[ToolResultFeatures]) -> list[float]:
        if not self.weights:
            raise ValueError("fit the scorer before prediction")
        return [self._sigmoid(sum(w * x for w, x in zip(self.weights, self._vector(f))))
                for f in features]

    def predict(self, features: Sequence[ToolResultFeatures], *, threshold: float = 0.5
                ) -> list[int]:
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        return [int(p >= threshold) for p in self.predict_proba(features)]


def feature_prune(messages: Sequence[Mapping[str, Any]], scorer: FeatureScorer, *,
                  threshold: float = 0.5, store: ContentStore | None = None,
                  protected_detector: Detector | None = None,
                  eligible_tools: Sequence[str] | None = None) -> PruneResult:
    features = extract_features(messages)
    labels = scorer.predict(list(features.values()), threshold=threshold)
    keep = {index for index, label in zip(features, labels) if label}
    return _mask(messages, keep, store=store, protected_detector=protected_detector,
                 eligible_tools=eligible_tools)
