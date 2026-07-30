import { Activity, Github } from "lucide-react";

export function Brand() {
  return (
    <header className="brand">
      <div className="brand-mark" aria-hidden="true">
        <Activity size={24} strokeWidth={1.8} />
      </div>
      <div>
        <p className="eyebrow">AGENT OPERATIONS</p>
        <h1>Admire's AI Harness</h1>
      </div>
      <a
        className="github-link"
        href="https://github.com/LittleChenLiya/admires-ai-harness"
        target="_blank"
        rel="noreferrer"
      >
        <Github size={16} /> Source
      </a>
    </header>
  );
}
