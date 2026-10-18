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

    # --- shot chain integrity -------------------------------------------
    # Execution order in ComfyUI is dependency-driven, so a shot only runs
    # after its first_frame source. Verifying the IMAGE chain therefore also
    # verifies the run order.
    for d in subs:
        insts = [n for n in wf["nodes"] if n["type"] == d["id"]]
        if len(insts) < 2:
            continue
        labels = [i.get("label", i["name"]) for i in d["inputs"]]
        if "first_frame" not in labels:
            continue
        ff = labels.index("first_frame")
        img_out = next((k for k, o in enumerate(d["outputs"])
                        if o["type"] == "IMAGE"), None)

        by_id = {n["id"]: n for n in wf["nodes"]}
        source = {}
        for raw in wf["links"]:
            l = norm(raw)
            if l["target_id"] in by_id and by_id[l["target_id"]]["type"] == d["id"] \
                    and l["target_slot"] == ff:
                source[l["target_id"]] = (l["origin_id"], l["origin_slot"])

        insts.sort(key=lambda n: n["pos"][0])          # left-to-right = shot order
        print()
        print("shot chain (left to right on canvas):")
        prev = None
        for k, n in enumerate(insts):
            src = source.get(n["id"])
            title = n.get("title", n["id"])
            if src is None:
                errors.append(f"[root] {title} has no first_frame input")
                desc = "UNCONNECTED"
            elif src[0] in by_id and by_id[src[0]]["type"] == d["id"]:
                sn = by_id[src[0]]
                desc = f"{sn.get('title', sn['id'])} output[{src[1]}]"
                if src[1] != img_out:
                    errors.append(f"[root] {title} seeded from output slot "
                                  f"{src[1]}, expected IMAGE slot {img_out}")
                if prev is not None and src[0] != prev["id"]:
                    errors.append(f"[root] {title} seeded from "
                                  f"{sn.get('title')}, not the immediately "
                                  f"preceding shot {prev.get('title')} — "
                                  f"chain order is broken")
            else:
                sn = by_id.get(src[0])
                desc = f"{sn.get('title', src[0]) if sn else src[0]} (external)"
                if k > 0:
                    notes.append(f"[root] {title} restarts from an external "
                                 f"source ({desc}) instead of continuing the "
                                 f"chain — intentional with anchor=reference")
            print(f"  {k+1}. {title:<10} first_frame ← {desc}")
            prev = n

    print()
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
