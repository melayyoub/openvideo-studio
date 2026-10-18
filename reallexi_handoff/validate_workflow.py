#!/usr/bin/env python3
"""Structural validator for ComfyUI workflow JSON (root graph + subgraph defs)."""
import json
import sys
from collections import Counter

errors, warns, notes = [], [], []


def norm(l):
    if isinstance(l, list):
        return dict(zip(("id", "origin_id", "origin_slot",
                         "target_id", "target_slot", "type"), l))
    return l


def check_graph(name, nodes, links, io_inputs=None, io_outputs=None,
                in_id=-10, out_id=-20):
    nd = {n["id"]: n for n in nodes}

    dup = [i for i, c in Counter(n["id"] for n in nodes).items() if c > 1]
    if dup:
        errors.append(f"[{name}] duplicate node ids: {dup}")

    ldup = [i for i, c in Counter(norm(l)["id"] for l in links).items() if c > 1]
    if ldup:
        errors.append(f"[{name}] duplicate link ids: {ldup}")

    seen_link_ids = set()
    for raw in links:
        l = norm(raw)
        lid = l["id"]
        seen_link_ids.add(lid)
        o, t = l["origin_id"], l["target_id"]

        if o not in nd and o != in_id:
            errors.append(f"[{name}] link {lid}: origin node {o} missing")
        if t not in nd and t != out_id:
            errors.append(f"[{name}] link {lid}: target node {t} missing")

        if o in nd:
            outs = nd[o].get("outputs") or []
            if l["origin_slot"] >= len(outs):
                errors.append(f"[{name}] link {lid}: origin slot "
                              f"{l['origin_slot']} out of range on "
                              f"{nd[o]['type']}({o}) which has {len(outs)}")
            elif lid not in (outs[l["origin_slot"]].get("links") or []):
                errors.append(f"[{name}] link {lid} not registered on origin "
                              f"{nd[o]['type']}({o}).outputs[{l['origin_slot']}]")
            elif outs[l["origin_slot"]]["type"] != l["type"]:
                warns.append(f"[{name}] link {lid} type {l['type']} != origin "
                             f"slot type {outs[l['origin_slot']]['type']}")

        if t in nd:
            ins = nd[t].get("inputs") or []
            if l["target_slot"] >= len(ins):
                errors.append(f"[{name}] link {lid}: target slot "
                              f"{l['target_slot']} out of range on "
                              f"{nd[t]['type']}({t}) which has {len(ins)}")
            elif ins[l["target_slot"]].get("link") != lid:
                errors.append(f"[{name}] link {lid} not registered on target "
                              f"{nd[t]['type']}({t}).inputs[{l['target_slot']}] "
                              f"(has {ins[l['target_slot']].get('link')})")

    # every socket-declared link must exist
    for n in nodes:
        for k, i in enumerate(n.get("inputs") or []):
            if i.get("link") is not None and i["link"] not in seen_link_ids:
                errors.append(f"[{name}] {n['type']}({n['id']}).inputs[{k}] "
                              f"references missing link {i['link']}")
        for k, o in enumerate(n.get("outputs") or []):
            for lk in (o.get("links") or []):
                if lk not in seen_link_ids:
                    errors.append(f"[{name}] {n['type']}({n['id']}).outputs[{k}] "
                                  f"references missing link {lk}")

    if io_inputs is not None:
        for spec in io_inputs:
            for lk in spec.get("linkIds", []):
                if lk not in seen_link_ids:
                    errors.append(f"[{name}] input '{spec['name']}' references "
                                  f"missing link {lk}")
                else:
                    l = [norm(x) for x in links if norm(x)["id"] == lk][0]
                    if l["origin_id"] != in_id:
                        errors.append(f"[{name}] input '{spec['name']}' link {lk} "
                                      f"does not originate from {in_id}")
        slots = [norm(x) for x in links if norm(x)["origin_id"] == in_id]
        declared = {lk for s in io_inputs for lk in s.get("linkIds", [])}
        for l in slots:
            if l["id"] not in declared:
                errors.append(f"[{name}] link {l['id']} from input node not "
                              f"declared in any input's linkIds")
            idx = l["origin_slot"]
            if idx >= len(io_inputs):
                errors.append(f"[{name}] link {l['id']} origin_slot {idx} "
                              f"exceeds input count {len(io_inputs)}")
            elif l["id"] not in io_inputs[idx].get("linkIds", []):
                errors.append(f"[{name}] link {l['id']} origin_slot {idx} points "
                              f"at input '{io_inputs[idx]['name']}' but is not in "
                              f"its linkIds")

    # ComfyUI resolves generic subgraph inputs by NAME, not array position.
    # Non-monotonic value_N numbering silently misroutes widgets and links.
    if io_inputs is not None:
        seq = []
        for idx, spec in enumerate(io_inputs):
            nm = spec.get("name", "")
            if nm == "value":
                seq.append((0, idx, spec.get("label", nm)))
            elif nm.startswith("value_") and nm.split("_")[-1].isdigit():
                seq.append((int(nm.split("_")[-1]), idx, spec.get("label", nm)))
        for a, b in zip(seq, seq[1:]):
            if b[0] != a[0] + 1 or b[1] < a[1]:
                errors.append(
                    f"[{name}] input numbering not monotonic: "
                    f"'{a[2]}'=value_{a[0]} at slot {a[1]}, then "
                    f"'{b[2]}'=value_{b[0]} at slot {b[1]}. ComfyUI maps these "
                    f"by name — widgets and links will land on wrong nodes.")

    if io_outputs is not None:
        for slot, spec in enumerate(io_outputs):
            for lk in spec.get("linkIds", []):
                match = [norm(x) for x in links if norm(x)["id"] == lk]
                if not match:
                    errors.append(f"[{name}] output '{spec['name']}' references "
                                  f"missing link {lk}")
                elif match[0]["target_id"] != out_id or match[0]["target_slot"] != slot:
                    errors.append(f"[{name}] output '{spec['name']}' link {lk} "
                                  f"mis-targeted")

    # orphan report
    touched = set()
    for raw in links:
        l = norm(raw)
        touched.add(l["origin_id"])
        touched.add(l["target_id"])
    orphans = [f"{n['type']}({n['id']})" for n in nodes
               if n["id"] not in touched and n["type"] not in
               ("MarkdownNote", "PrimitiveStringMultiline", "Note")]
    if orphans:
        notes.append(f"[{name}] unconnected (may be intentional): {orphans}")


def main(path):
    wf = json.loads(open(path).read())
    check_graph("root", wf["nodes"], wf["links"])

    subs = wf.get("definitions", {}).get("subgraphs", [])
    sub_ids = {s["id"] for s in subs}
    for s in subs:
        check_graph(f"sub:{s['name']}", s["nodes"], s["links"],
                    s.get("inputs"), s.get("outputs"),
                    s["inputNode"]["id"], s["outputNode"]["id"])

    # instance-node contract
    by_id = {s["id"]: s for s in subs}
    for n in wf["nodes"]:
        if n["type"] in sub_ids:
            d = by_id[n["type"]]
            dnames = [i["name"] for i in d["inputs"]]
            inames = [i["name"] for i in n.get("inputs", [])]
            if inames != dnames:
                errors.append(f"[root] instance {n['id']} input names {inames} "
                              f"!= definition {dnames}")
            wcount = sum(1 for i in d["inputs"]
                         if any(t.get("widget") for t in _targets(d, i)))
            got = len(n.get("widgets_values", []))
            if got != wcount:
                errors.append(f"[root] instance {n['id']} has {got} "
                              f"widgets_values but {wcount} widget-backed inputs")
            donames = [o["name"] for o in d["outputs"]]
            gonames = [o["name"] for o in n.get("outputs", [])]
            if gonames != donames:
                errors.append(f"[root] instance {n['id']} outputs {gonames} "
                              f"!= definition {donames}")

    print(f"nodes(root)={len(wf['nodes'])} links(root)={len(wf['links'])} "
          f"groups(root)={len(wf['groups'])} subgraphs={len(subs)}")
    for s in subs:
        print(f"  · {s['name']}: {len(s['nodes'])} nodes, {len(s['links'])} links, "
              f"{len(s['groups'])} groups, {len(s['inputs'])} in, "
              f"{len(s['outputs'])} out")
    print()
    for e in errors:
        print("ERROR  ", e)
    for w in warns:
        print("WARN   ", w)
    for nt in notes:
        print("NOTE   ", nt)
    print()
    print("RESULT:", "FAIL" if errors else "PASS")
    return 1 if errors else 0


def _targets(defn, inp):
    """Return the input-socket dicts inside the definition fed by this SG input."""
    out = []
    nd = {n["id"]: n for n in defn["nodes"]}
    for raw in defn["links"]:
        l = norm(raw)
        if l["id"] in inp.get("linkIds", []) and l["target_id"] in nd:
            ins = nd[l["target_id"]].get("inputs") or []
            if l["target_slot"] < len(ins):
                out.append(ins[l["target_slot"]])
    return out


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
