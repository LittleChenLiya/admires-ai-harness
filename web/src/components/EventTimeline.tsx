import { Check, ChevronDown, Circle, X } from "lucide-react";
import { useState } from "react";

import type { RunEvent } from "../types";

function iconFor(type: string) {
  if (type.endsWith("failed")) return <X size={13} />;
  if (type.endsWith("completed") || type.endsWith("saved"))
    return <Check size={13} />;
  return <Circle size={9} fill="currentColor" />;
}

function summary(event: RunEvent): string {
  const payload = event.payload;
  return String(
    payload.message ??
      payload.name ??
      payload.status ??
      payload.tool ??
      event.node_id ??
      "",
  );
}

export function EventTimeline({ events }: { events: RunEvent[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  return (
    <div className="timeline" aria-live="polite">
      {events.length === 0 ? (
        <div className="empty">
          Structured events will appear here in real time.
        </div>
      ) : (
        events.map((event) => {
          const isExpanded = expanded === event.event_id;
          return (
            <div className="event" key={event.event_id}>
              <span
                className={`event-icon ${event.type.endsWith("failed") ? "danger" : ""}`}
              >
                {iconFor(event.type)}
              </span>
              <button
                type="button"
                className="event-content"
                onClick={() => setExpanded(isExpanded ? null : event.event_id)}
                aria-expanded={isExpanded}
              >
                <span className="event-sequence">#{event.sequence}</span>
                <strong>{event.type}</strong>
                <span className="event-summary">{summary(event)}</span>
                <time>{new Date(event.timestamp).toLocaleTimeString()}</time>
                <ChevronDown size={13} className={isExpanded ? "rotate" : ""} />
              </button>
              {isExpanded && (
                <pre>{JSON.stringify(event.payload, null, 2)}</pre>
              )}
            </div>
          );
        })
      )}
    </div>
  );
}
