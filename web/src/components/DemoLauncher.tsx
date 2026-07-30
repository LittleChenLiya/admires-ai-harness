import { ArrowRight, Boxes, Code2, Search } from "lucide-react";
import { useMemo, useState } from "react";

import type { Demo } from "../types";

const icons = { research: Search, coding: Code2, collaboration: Boxes };

interface Props {
  demos: Demo[];
  busy: boolean;
  onLaunch: (demo: Demo, input: string) => Promise<void>;
}

export function DemoLauncher({ demos, busy, onLaunch }: Props) {
  const [selected, setSelected] = useState<string>(demos[0]?.id ?? "research");
  const demo = useMemo(
    () => demos.find((candidate) => candidate.id === selected) ?? demos[0],
    [demos, selected],
  );
  const [inputOverrides, setInputOverrides] = useState<Record<string, string>>(
    {},
  );

  if (!demo) return <div className="empty">No demos are registered.</div>;
  const input = inputOverrides[demo.id] ?? demo.defaultInput;

  return (
    <section className="launcher panel" aria-labelledby="launcher-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">OFFLINE-FIRST</p>
          <h2 id="launcher-title">Start a reference workflow</h2>
        </div>
        <span className="mock-badge">Mock model · $0.00</span>
      </div>
      <div
        className="demo-tabs"
        role="tablist"
        aria-label="Reference workflows"
      >
        {demos.map((item) => {
          const Icon = icons[item.id as keyof typeof icons] ?? Boxes;
          return (
            <button
              type="button"
              role="tab"
              aria-selected={item.id === demo.id}
              className={item.id === demo.id ? "demo-tab active" : "demo-tab"}
              key={item.id}
              onClick={() => setSelected(item.id)}
            >
              <Icon size={18} />
              <span>{item.title}</span>
            </button>
          );
        })}
      </div>
      <p className="demo-description">{demo.description}</p>
      <div className="capabilities">
        {demo.capabilities.map((capability) => (
          <span key={capability}>{capability}</span>
        ))}
      </div>
      <label className="input-label" htmlFor="run-input">
        Task input
      </label>
      <textarea
        id="run-input"
        value={input}
        rows={3}
        onChange={(event) =>
          setInputOverrides((current) => ({
            ...current,
            [demo.id]: event.target.value,
          }))
        }
      />
      <button
        className="launch-button"
        type="button"
        disabled={busy || input.trim().length === 0}
        onClick={() => void onLaunch(demo, input)}
      >
        {busy ? "Starting…" : "Run workflow"}
        <ArrowRight size={17} />
      </button>
    </section>
  );
}
